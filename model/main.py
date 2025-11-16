import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras.layers import StringLookup
from keras import ops
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn
import shutil
import uuid
from datetime import datetime
import json

# Глобальные переменные для кэширования модели распознавания
_PREDICTION_MODEL = None
_CHAR_TO_NUM = None
_NUM_TO_CHAR = None

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32

# Настройки FastAPI
app = FastAPI(
    title="Handwritten Text Recognition API",
    description="API для сегментации и распознавания рукописного русского текста",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для временных файлов
TEMP_DIR = "temp_uploads"
RESULTS_DIR = "api_results"

def initialize_directories():
    """Создание необходимых директорий"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def initialize_recognition_model():
    """Инициализация модели распознавания текста"""
    global _PREDICTION_MODEL, _CHAR_TO_NUM, _NUM_TO_CHAR
    
    if _PREDICTION_MODEL is not None:
        return
    
    try:
        characters = sorted(list(['!', '"', '%', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'R', '[', ']', 'b', 'c', 'e', 'h', 'i', 'o', 'p', 'r', 's', 't', 'u', '«', '»', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё', '№']))

        _CHAR_TO_NUM = StringLookup(vocabulary=list(characters), mask_token=None)
        _NUM_TO_CHAR = StringLookup(
            vocabulary=_CHAR_TO_NUM.get_vocabulary(), mask_token=None, invert=True
        )
        
        class CTCLayer(keras.layers.Layer):
            def __init__(self, padding_token, name=None, *args, **kwargs):
                super().__init__(name=name, **kwargs)
                self.loss_fn = tf.keras.backend.ctc_batch_cost
                self.padding_token = padding_token

            def call(self, y_true, y_pred, **kwargs):
                batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
                time_steps = ops.cast(ops.shape(y_pred)[1], dtype="int64")
                input_length = time_steps * ops.ones(shape=(batch_len, 1), dtype="int64")
                mask = tf.math.not_equal(y_true, self.padding_token)
                label_length = ops.cast(tf.reduce_sum(ops.cast(mask, dtype="int32"), axis=1), dtype="int64")
                label_length = ops.reshape(label_length, (batch_len, 1))
                loss = self.loss_fn(y_true, y_pred, input_length, label_length)
                self.add_loss(loss)
                return y_pred

        base_model = keras.models.load_model(
            "model.h5",
            custom_objects={"CTCLayer": CTCLayer},
            compile=False
        )

        _PREDICTION_MODEL = keras.Model(
            base_model.get_layer("image").output,
            base_model.get_layer("output").output
        )
        print("✅ Модель распознавания загружена успешно")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        raise

def distortion_free_resize(image, img_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    """Изменение размера изображения без искажений"""
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    pad_height = h - ops.shape(image)[0]
    pad_width = w - ops.shape(image)[1]
    pad_height_top = pad_height // 2 + (pad_height % 2)
    pad_height_bottom = pad_height // 2
    pad_width_left = pad_width // 2 + (pad_width % 2)
    pad_width_right = pad_width // 2
    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    image = ops.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image_cv2(img_path):
    """Предобработка изображения для распознавания"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Не удалось загрузить изображение " + img_path)
    img = np.expand_dims(img, axis=-1)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = distortion_free_resize(img)
    img = img / 255.0
    return img

def decode_batch(pred):
    """Декодирование предсказания модели"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.ops.nn.ctc_decode(pred, sequence_lengths=input_len)[0][0][:, :16]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(res != -1))
        res = (
            tf.strings.reduce_join(_NUM_TO_CHAR(res))
            .numpy()
            .decode("utf-8")
            .replace("[UNK]", "")
        )
        output_text.append(res)
    return output_text

def predict_image(img_path):
    """Распознавание текста на изображении"""
    initialize_recognition_model()
    
    img = preprocess_image_cv2(img_path)
    img = tf.expand_dims(img, 0)
    preds = _PREDICTION_MODEL.predict(img, verbose=0)
    text = decode_batch(preds)[0]
    return text

# Функции сегментации текста
def preprocess_image_segmentation(image_path):
    """Предварительная обработка изображения для сегментации"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # === ИЗМЕНЕНИЕ: расширяем (dilate) изображение, чтобы объединить близкие буквы ===
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # ← ИЗМЕНЕНО
    binary = cv2.dilate(binary, kernel, iterations=1)            # ← ИЗМЕНЕНО
    
    return img, binary

def remove_noise(binary_img):
    """Удаление шума"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    return cleaned

def expand_bbox(bbox, expansion=5, img_shape=None):
    """Расширение bounding box"""
    x, y, w, h = bbox
    x_new = max(0, x - expansion)
    y_new = max(0, y - expansion)
    w_new = w + 2 * expansion
    h_new = h + 2 * expansion
    
    if img_shape is not None:
        height, width = img_shape[:2]
        if x_new + w_new > width:
            w_new = width - x_new
        if y_new + h_new > height:
            h_new = height - y_new
    
    return (x_new, y_new, w_new, h_new)

def is_punctuation(bbox, binary_img, min_height_ratio=0.15, max_aspect_ratio=1.5):
    """Определение знаков препинания"""
    x, y, w, h = bbox
    aspect_ratio = w / h
    height_ratio = h / binary_img.shape[0]
    
    is_narrow = aspect_ratio < max_aspect_ratio
    is_small = height_ratio < min_height_ratio
    
    roi = binary_img[y:y+h, x:x+w]
    if roi.size > 0:
        black_pixels = np.sum(roi == 255)
        total_pixels = roi.size
        density = black_pixels / total_pixels
        is_dense = density > 0.6
    else:
        is_dense = False
    
    return is_small and is_narrow and is_dense

def find_words(binary_img, original_img, min_word_width=20, min_word_height=15):
    """Поиск слов на изображении с правильной сортировкой для многострочного текста"""
    result_img = original_img.copy()
    height, width = original_img.shape[:2]
    
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    words = []
    word_bboxes = []
    punctuation_bboxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        expanded_bbox = expand_bbox((x, y, w, h), expansion=3, img_shape=original_img.shape)
        x, y, w, h = expanded_bbox
        
        if w >= min_word_width and h >= min_word_height:
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            bbox_area = w * h
            fill_ratio = area / bbox_area if bbox_area > 0 else 0
            
            if fill_ratio > 0.1 and 0.3 < aspect_ratio < 8:
                if is_punctuation((x, y, w, h), binary_img):
                    punctuation_bboxes.append((x, y, w, h))
                else:
                    words.append(contour)
                    word_bboxes.append((x, y, w, h))
    
    # === ИСПРАВЛЕНИЕ: правильная сортировка для многострочного текста ===
    # Сначала группируем bbox по строкам, затем сортируем каждую строку
    word_bboxes = sort_bboxes_multi_line(word_bboxes, line_threshold=0.5)
    
    final_bboxes = []
    for bbox in word_bboxes:
        expanded_bbox = expand_bbox(bbox, expansion=5, img_shape=original_img.shape)
        final_bboxes.append(expanded_bbox)
    
    # Отрисовка bounding boxes для слов
    for i, (x, y, w, h) in enumerate(final_bboxes):
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_img, f'Word {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Отрисовка знаков препинания
    for i, (x, y, w, h) in enumerate(punctuation_bboxes):
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(result_img, 'Punct', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    return result_img, final_bboxes, len(punctuation_bboxes)

def sort_bboxes_multi_line(bboxes, line_threshold=0.5):
    """
    Сортировка bounding boxes для многострочного текста:
    сначала по строкам (сверху вниз), затем по горизонтали (слева направо)
    
    Args:
        bboxes: список bounding boxes в формате (x, y, w, h)
        line_threshold: порог для определения принадлежности к одной строке (0.0-1.0)
                        относительно средней высоты строки
    """
    if not bboxes:
        return []
    
    # Вычисляем среднюю высоту bbox
    avg_height = sum(h for _, _, _, h in bboxes) / len(bboxes)
    line_threshold_px = avg_height * line_threshold
    
    # Группируем bbox по строкам
    lines = []
    
    for bbox in bboxes:
        x, y, w, h = bbox
        center_y = y + h / 2
        
        # Ищем подходящую строку
        found_line = False
        for line in lines:
            # Проверяем, принадлежит ли bbox к этой строке
            line_center_y = line['center_y']
            if abs(center_y - line_center_y) <= line_threshold_px:
                line['bboxes'].append(bbox)
                # Обновляем центр строки
                line['center_y'] = sum(by + bh/2 for bx, by, bw, bh in line['bboxes']) / len(line['bboxes'])
                found_line = True
                break
        
        # Если не нашли подходящую строку, создаем новую
        if not found_line:
            lines.append({
                'center_y': center_y,
                'bboxes': [bbox]
            })
    
    # Сортируем строки по вертикали (сверху вниз)
    lines.sort(key=lambda line: line['center_y'])
    
    # Сортируем bbox в каждой строке по горизонтали (слева направо)
    sorted_bboxes = []
    for line in lines:
        line_bboxes = sorted(line['bboxes'], key=lambda bbox: bbox[0])
        sorted_bboxes.extend(line_bboxes)
    
    return sorted_bboxes

def merge_close_bboxes(bboxes, max_gap=50, max_vertical_gap=20):
    """Объединение близких bounding boxes с учетом многострочности"""
    if not bboxes:
        return []
    
    # Сначала сортируем bbox для правильной обработки строк
    sorted_bboxes = sort_bboxes_multi_line(bboxes)
    
    merged = []
    current_bbox = list(sorted_bboxes[0])
    
    for bbox in sorted_bboxes[1:]:
        x, y, w, h = bbox
        current_x, current_y, current_w, current_h = current_bbox
        
        # Проверяем, находятся ли bbox на одной строке
        current_center_y = current_y + current_h / 2
        bbox_center_y = y + h / 2
        same_line = abs(current_center_y - bbox_center_y) < (current_h * 0.5)
        
        if same_line:
            horizontal_gap = x - (current_x + current_w)
            vertical_overlap = min(current_y + current_h, y + h) - max(current_y, y)
            
            if horizontal_gap <= max_gap and vertical_overlap >= -max_vertical_gap:
                new_x = min(current_x, x)
                new_y = min(current_y, y)
                new_w = max(current_x + current_w, x + w) - new_x
                new_h = max(current_y + current_h, y + h) - new_y
                current_bbox = [new_x, new_y, new_w, new_h]
            else:
                merged.append(tuple(current_bbox))
                current_bbox = list(bbox)
        else:
            # Разные строки - не объединяем
            merged.append(tuple(current_bbox))
            current_bbox = list(bbox)
    
    merged.append(tuple(current_bbox))
    return merged

def process_image_segmentation(image_path, output_dir):
    """Обработка изображения: сегментация и распознавание"""
    # Создаем уникальную папку для результатов
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(output_dir, session_id)
    words_dir = os.path.join(session_dir, "words")
    os.makedirs(words_dir, exist_ok=True)
    
    # Предварительная обработка
    original_img, binary_img = preprocess_image_segmentation(image_path)
    
    # Удаление шума
    cleaned_img = remove_noise(binary_img)
    
    # Поиск слов
    result_img, word_bboxes, punctuation_count = find_words(cleaned_img, original_img)
    
    # Сохранение результата с bounding boxes
    result_image_path = os.path.join(session_dir, "segmentation_result.jpg")
    cv2.imwrite(result_image_path, result_img)
    
    # Сохранение и распознавание отдельных слов
    recognition_results = []
    
    for i, (x, y, w, h) in enumerate(word_bboxes):
        try:
            # Вырезаем слово из изображения
            padding = 5
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(original_img.shape[1], x + w + padding)
            y_end = min(original_img.shape[0], y + h + padding)
            
            word_img = original_img[y_start:y_end, x_start:x_end]
            
            if word_img.size > 0:
                # Сохраняем изображение слова
                filename = f"word_{i+1:03d}.png"
                filepath = os.path.join(words_dir, filename)
                cv2.imwrite(filepath, word_img)
                
                # Распознаем текст на изображении
                recognized_text = predict_image(filepath)
                
                recognition_results.append({
                    'word_number': i + 1,
                    'filename': filename,
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'recognized_text': recognized_text,
                    'filepath': filepath
                })
                
        except Exception as e:
            recognition_results.append({
                'word_number': i + 1,
                'filename': f"word_{i+1:03d}.png",
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'recognized_text': f"ОШИБКА: {str(e)}",
                'filepath': ""
            })
    
    # Формируем полный текст
    full_text = " ".join([result['recognized_text'] for result in recognition_results])
    
    # Сохраняем результаты в JSON
    results_data = {
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            'total_words': len(word_bboxes),
            'punctuation_marks': punctuation_count,
            'recognized_words': len(recognition_results)
        },
        'recognition_results': recognition_results,
        'full_text': full_text,
        'result_image': result_image_path,
        'words_directory': words_dir
    }
    
    results_json_path = os.path.join(session_dir, "results.json")
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    return results_data

# API endpoints
@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Handwritten Text Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "recognize": "/recognize (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка состояния API"""
    try:
        initialize_recognition_model()
        return {
            "status": "healthy",
            "model_loaded": _PREDICTION_MODEL is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

@app.post("/recognize")
async def recognize_text(
    file: UploadFile = File(..., description="Изображение с рукописным текстом (JPG, PNG)"),
    return_image: bool = False
):
    """
    Распознавание рукописного текста на изображении
    
    - **file**: Изображение с рукописным текстом
    - **return_image**: Возвращать ли изображение с разметкой
    """
    # Проверка типа файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    # Создаем уникальное имя файла
    file_extension = os.path.splitext(file.filename)[1] or '.jpg'
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_filepath = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        # Сохраняем загруженный файл
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Обрабатываем изображение
        results = process_image_segmentation(temp_filepath, RESULTS_DIR)
        
        # Формируем ответ
        response_data = {
            "session_id": results['session_id'],
            "timestamp": results['timestamp'],
            "statistics": results['statistics'],
            "full_text": results['full_text'],
            "words": [
                {
                    "word_number": word['word_number'],
                    "text": word['recognized_text'],
                    "bbox": word['bbox']
                }
                for word in results['recognition_results']
            ]
        }
        
        # Добавляем URL для скачивания изображения если требуется
        if return_image:
            response_data["result_image_url"] = f"/download/{results['session_id']}/image"
            response_data["words_images_url"] = f"/download/{results['session_id']}/words"
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

@app.get("/download/{session_id}/image")
async def download_result_image(session_id: str):
    """Скачать изображение с разметкой сегментации"""
    image_path = os.path.join(RESULTS_DIR, session_id, "segmentation_result.jpg")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    
    return FileResponse(
        image_path,
        media_type='image/jpeg',
        filename=f"segmentation_result_{session_id}.jpg"
    )

@app.get("/download/{session_id}/words")
async def download_words_archive(session_id: str):
    """Скачать архив с изображениями слов"""
    words_dir = os.path.join(RESULTS_DIR, session_id, "words")
    if not os.path.exists(words_dir):
        raise HTTPException(status_code=404, detail="Слова не найдены")
    
    # Создаем архив
    archive_path = os.path.join(RESULTS_DIR, session_id, f"words_{session_id}.zip")
    shutil.make_archive(
        os.path.splitext(archive_path)[0],
        'zip',
        words_dir
    )
    
    return FileResponse(
        archive_path,
        media_type='application/zip',
        filename=f"words_{session_id}.zip"
    )

@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """Получить результаты обработки по session_id"""
    results_path = os.path.join(RESULTS_DIR, session_id, "results.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Результаты не найдены")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    return JSONResponse(content=results_data)

# Инициализация при запуске
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    initialize_directories()
    try:
        initialize_recognition_model()
        print("🚀 API запущен и готов к работе!")
        print("📚 Документация доступна по адресу: http://localhost:8000/docs")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")

# Запуск приложения
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )