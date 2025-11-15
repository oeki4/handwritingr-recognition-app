import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def preprocess_image(image_path):
    """
    Предварительная обработка изображения
    """
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Конвертация в grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Увеличение контраста с помощью CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Бинаризация (адаптивный порог)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return img, binary

def remove_noise(binary_img):
    """
    Удаление шума с помощью морфологических операций
    """
    # Создание ядра для морфологических операций
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    # Открытие для удаления мелкого шума
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    
    # Закрытие для соединения близких элементов
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_closing)
    
    return cleaned

def expand_bbox(bbox, expansion=5, img_shape=None):
    """
    Расширение bounding box с проверкой границ изображения
    """
    x, y, w, h = bbox
    
    # Расширяем bounding box
    x_new = max(0, x - expansion)
    y_new = max(0, y - expansion)
    w_new = w + 2 * expansion
    h_new = h + 2 * expansion
    
    # Проверяем границы изображения
    if img_shape is not None:
        height, width = img_shape[:2]
        if x_new + w_new > width:
            w_new = width - x_new
        if y_new + h_new > height:
            h_new = height - y_new
    
    return (x_new, y_new, w_new, h_new)

def is_punctuation(bbox, binary_img, min_height_ratio=0.15, max_aspect_ratio=1.5):
    """
    Определение, является ли bounding box знаком препинания
    """
    x, y, w, h = bbox
    
    # Вычисляем характеристики
    aspect_ratio = w / h
    height_ratio = h / binary_img.shape[0]
    
    # Знаки препинания обычно узкие и высокие или очень маленькие
    is_narrow = aspect_ratio < max_aspect_ratio
    is_small = height_ratio < min_height_ratio
    
    # Дополнительная проверка: анализируем область внутри bounding box
    roi = binary_img[y:y+h, x:x+w]
    if roi.size > 0:
        # Вычисляем плотность черных пикселей
        black_pixels = np.sum(roi == 255)
        total_pixels = roi.size
        density = black_pixels / total_pixels
        
        # Знаки препинания обычно имеют высокую плотность черных пикселей
        is_dense = density > 0.6
    else:
        is_dense = False
    
    # Считаем знаком препинания если объект маленький И узкий И плотный
    return is_small and is_narrow and is_dense

def find_words(binary_img, original_img, min_word_width=20, min_word_height=15):
    """
    Поиск и выделение слов на изображении
    """
    # Копия оригинального изображения для рисования
    result_img = original_img.copy()
    height, width = original_img.shape[:2]
    
    # Поиск контуров
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    words = []
    word_bboxes = []
    punctuation_bboxes = []
    
    for contour in contours:
        # Получение ограничивающего прямоугольника
        x, y, w, h = cv2.boundingRect(contour)
        
        # Расширяем bounding box чтобы захватить края букв
        expanded_bbox = expand_bbox((x, y, w, h), expansion=3, img_shape=original_img.shape)
        x, y, w, h = expanded_bbox
        
        # Фильтрация по размеру (удаление слишком маленьких объектов)
        if w >= min_word_width and h >= min_word_height:
            # Фильтрация по соотношению сторон
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            bbox_area = w * h
            
            # Проверяем, что контур заполняет достаточную площадь bounding box
            fill_ratio = area / bbox_area if bbox_area > 0 else 0
            
            if fill_ratio > 0.1 and 0.3 < aspect_ratio < 8:
                # Проверяем, не является ли объект знаком препинания
                if is_punctuation((x, y, w, h), binary_img):
                    punctuation_bboxes.append((x, y, w, h))
                else:
                    words.append(contour)
                    word_bboxes.append((x, y, w, h))
    
    # Сортировка слов слева направо
    word_bboxes.sort(key=lambda bbox: bbox[0])
    
    # Объединение близко расположенных bounding boxes
    merged_bboxes = merge_close_bboxes(word_bboxes, max_gap=20, max_vertical_gap=10)
    
    # Расширяем итоговые bounding boxes для слов
    final_bboxes = []
    for bbox in merged_bboxes:
        expanded_bbox = expand_bbox(bbox, expansion=5, img_shape=original_img.shape)
        final_bboxes.append(expanded_bbox)
    
    # Отрисовка bounding boxes для слов (зеленые)
    for i, (x, y, w, h) in enumerate(final_bboxes):
        # Рисование прямоугольника вокруг слова
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Добавление номера слова
        cv2.putText(result_img, f'Word {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Отрисовка bounding boxes для знаков препинания (красные)
    for i, (x, y, w, h) in enumerate(punctuation_bboxes):
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(result_img, 'Punct', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    print(f"Найдено знаков препинания: {len(punctuation_bboxes)}")
    
    return result_img, final_bboxes

def merge_close_bboxes(bboxes, max_gap=20, max_vertical_gap=10):
    """
    Объединение близко расположенных bounding boxes
    """
    if not bboxes:
        return []
    
    # Сортировка по x-координате
    bboxes.sort(key=lambda bbox: bbox[0])
    
    merged = []
    current_bbox = list(bboxes[0])
    
    for bbox in bboxes[1:]:
        x, y, w, h = bbox
        current_x, current_y, current_w, current_h = current_bbox
        
        # Вычисляем горизонтальный и вертикальный зазоры
        horizontal_gap = x - (current_x + current_w)
        vertical_overlap = min(current_y + current_h, y + h) - max(current_y, y)
        
        # Объединяем если горизонтальный зазор маленький И есть вертикальное перекрытие
        if horizontal_gap <= max_gap and vertical_overlap >= -max_vertical_gap:
            # Объединение bounding boxes
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

def save_individual_words(original_img, bboxes, output_dir="words_output"):
    """
    Сохранение каждого слова как отдельного изображения
    """
    # Создаем папку если не существует
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Очищаем папку от предыдущих результатов (опционально)
    for file in Path(output_dir).glob("word_*.png"):
        file.unlink()
    
    saved_files = []
    
    for i, (x, y, w, h) in enumerate(bboxes):
        try:
            # Вырезаем слово из изображения с небольшим запасом
            padding = 5
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(original_img.shape[1], x + w + padding)
            y_end = min(original_img.shape[0], y + h + padding)
            
            word_img = original_img[y_start:y_end, x_start:x_end]
            
            # Проверяем что изображение не пустое
            if word_img.size > 0:
                # Создаем имя файла с нумерацией
                filename = f"word_{i+1:03d}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Сохраняем изображение
                cv2.imwrite(filepath, word_img)
                saved_files.append(filepath)
                print(f"✓ Сохранено: {filename} ({w}x{h} пикселей)")
            else:
                print(f"✗ Пустое изображение для слова {i+1}")
                
        except Exception as e:
            print(f"✗ Ошибка при сохранении слова {i+1}: {e}")
    
    return saved_files

def segment_words(image_path, output_path=None, words_output_dir="words_output"):
    """
    Основная функция для сегментации текста на слова
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
    
    # Сохранение отдельных слов
    saved_files = save_individual_words(original_img, word_bboxes, words_output_dir)
    
    # Вывод информации
    print(f"\n📊 Статистика:")
    print(f"Найдено слов: {len(word_bboxes)}")
    print(f"Сохранено файлов: {len(saved_files)}")
    print(f"Папка с словами: {words_output_dir}")
    
    for i, (x, y, w, h) in enumerate(word_bboxes):
        print(f"Слово {i+1}: позиция ({x},{y}), размер {w}x{h}")
    
    return result_img, word_bboxes, saved_files

def visualize_results(original_img, binary_img, result_img):
    """
    Визуализация промежуточных результатов
    """
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
    image_path = "stroke1.jpg"
    output_path = "output.jpg"
    words_output_dir = "extracted_words"  # Папка для сохранения слов
    
    try:
        print("🚀 Запуск сегментации рукописного текста...")
        
        # Выполнение сегментации
        result_img, word_bboxes, saved_files = segment_words(
            image_path, 
            output_path, 
            words_output_dir
        )
        
        print(f"\n✅ Сегментация завершена успешно!")
        print(f"📂 Слова сохранены в папке: {words_output_dir}")
        
        # Дополнительная визуализация (опционально)
        show_visualization = input("Показать визуализацию результатов? (y/n): ").lower()
        if show_visualization == 'y':
            original_img, binary_img = preprocess_image(image_path)
            cleaned_img = remove_noise(binary_img)
            visualize_results(original_img, cleaned_img, result_img)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")