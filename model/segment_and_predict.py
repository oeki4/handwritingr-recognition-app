import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras.layers import StringLookup
from keras import ops

# Глобальные переменные для кэширования модели распознавания
_PREDICTION_MODEL = None
_CHAR_TO_NUM = None
_NUM_TO_CHAR = None

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32

def initialize_recognition_model():
    """Инициализация модели распознавания текста"""
    global _PREDICTION_MODEL, _CHAR_TO_NUM, _NUM_TO_CHAR
    
    if _PREDICTION_MODEL is not None:
        return
    
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
        "checkpoint_best.h5",
        custom_objects={"CTCLayer": CTCLayer},
        compile=False
    )

    _PREDICTION_MODEL = keras.Model(
        base_model.get_layer("image").output,
        base_model.get_layer("output").output
    )

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

# Функции сегментации текста (остаются без изменений)
def preprocess_image(image_path):
    """Предварительная обработка изображения для сегментации"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return img, binary

def remove_noise(binary_img):
    """Удаление шума"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_closing)
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
    """Поиск слов на изображении"""
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
    
    word_bboxes.sort(key=lambda bbox: bbox[0])
    merged_bboxes = merge_close_bboxes(word_bboxes, max_gap=20, max_vertical_gap=10)
    
    final_bboxes = []
    for bbox in merged_bboxes:
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
    
    print(f"Найдено знаков препинания: {len(punctuation_bboxes)}")
    
    return result_img, final_bboxes

def merge_close_bboxes(bboxes, max_gap=20, max_vertical_gap=10):
    """Объединение близких bounding boxes"""
    if not bboxes:
        return []
    
    bboxes.sort(key=lambda bbox: bbox[0])
    merged = []
    current_bbox = list(bboxes[0])
    
    for bbox in bboxes[1:]:
        x, y, w, h = bbox
        current_x, current_y, current_w, current_h = current_bbox
        
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
    
    merged.append(tuple(current_bbox))
    return merged

def save_and_recognize_words(original_img, bboxes, output_dir="words_output"):
    """
    Сохранение каждого слова как отдельного изображения и распознавание текста
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Очищаем папку от предыдущих результатов
    for file in Path(output_dir).glob("word_*.png"):
        file.unlink()
    
    recognition_results = []
    
    for i, (x, y, w, h) in enumerate(bboxes):
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
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, word_img)
                
                # Распознаем текст на изображении
                recognized_text = predict_image(filepath)
                
                recognition_results.append({
                    'word_number': i + 1,
                    'filename': filename,
                    'bbox': (x, y, w, h),
                    'recognized_text': recognized_text,
                    'filepath': filepath
                })
                
                print(f"✓ Слово {i+1:03d}: '{recognized_text}'")
                
            else:
                print(f"✗ Пустое изображение для слова {i+1}")
                
        except Exception as e:
            print(f"✗ Ошибка при обработке слова {i+1}: {e}")
            recognition_results.append({
                'word_number': i + 1,
                'filename': f"word_{i+1:03d}.png",
                'bbox': (x, y, w, h),
                'recognized_text': f"ОШИБКА: {str(e)}",
                'filepath': ""
            })
    
    return recognition_results

def segment_and_recognize_words(image_path, output_path=None, words_output_dir="words_output"):
    """
    Основная функция для сегментации текста на слова и их распознавания
    """
    # Предварительная обработка
    original_img, binary_img = preprocess_image(image_path)
    
    # Удаление шума
    cleaned_img = remove_noise(binary_img)
    
    # Поиск слов
    result_img, word_bboxes = find_words(cleaned_img, original_img)
    
    # Сохранение общего результата с bounding boxes
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"📁 Результат с bounding boxes сохранен в: {output_path}")
    
    # Сохранение и распознавание отдельных слов
    print("\n🔍 Начинаю распознавание слов...")
    recognition_results = save_and_recognize_words(original_img, word_bboxes, words_output_dir)
    
    # Вывод итоговых результатов
    print(f"\n📊 ИТОГИ РАСПОЗНАВАНИЯ:")
    print(f"Найдено слов: {len(word_bboxes)}")
    print(f"Распознано слов: {len(recognition_results)}")
    print(f"Папка с словами: {words_output_dir}")
    
    print(f"\n📝 РАСПОЗНАННЫЙ ТЕКСТ:")
    full_text = ""
    for result in recognition_results:
        print(f"Слово {result['word_number']:03d}: '{result['recognized_text']}'")
        full_text += result['recognized_text'] + " "
    
    print(f"\n📄 ПОЛНЫЙ ТЕКСТ:")
    print(full_text.strip())
    
    # Сохраняем результаты в текстовый файл
    results_file = os.path.join(words_output_dir, "recognition_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ ТЕКСТА\n")
        f.write("=" * 50 + "\n")
        for result in recognition_results:
            f.write(f"Слово {result['word_number']:03d}: '{result['recognized_text']}'\n")
        f.write("\nПОЛНЫЙ ТЕКСТ:\n")
        f.write(full_text.strip())
    
    print(f"📄 Результаты сохранены в: {results_file}")
    
    return result_img, word_bboxes, recognition_results, full_text.strip()

def visualize_results(original_img, binary_img, result_img):
    """Визуализация результатов"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Бинаризованное изображение')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Результат сегментации\n(Зеленые - слова, Красные - знаки препинания)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Пример использования
if __name__ == "__main__":
    # Замените путь на путь к вашему изображению
    image_path = "stroke5.jpg"
    output_path = "output.jpg"
    words_output_dir = "extracted_words"
    
    try:
        print("🚀 Запуск сегментации и распознавания рукописного текста...")
        
        # Выполнение сегментации и распознавания
        result_img, word_bboxes, recognition_results, full_text = segment_and_recognize_words(
            image_path, 
            output_path, 
            words_output_dir
        )
        
        print(f"\n✅ Обработка завершена успешно!")
        print(f"📂 Слова сохранены в папке: {words_output_dir}")
        print(f"📄 Полный распознанный текст: {full_text}")
        
        # Дополнительная визуализация (опционально)
        show_visualization = input("\nПоказать визуализацию результатов? (y/n): ").lower()
        if show_visualization == 'y':
            original_img, binary_img = preprocess_image(image_path)
            cleaned_img = remove_noise(binary_img)
            visualize_results(original_img, cleaned_img, result_img)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")