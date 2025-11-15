import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import StringLookup
from keras import ops
import cv2
import pandas as pd
import os

# Оптимизации
tf.config.optimizer.set_jit(True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Глобальные переменные для кэширования
_PREDICTION_MODEL = None
_CHAR_TO_NUM = None
_NUM_TO_CHAR = None

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32

def initialize_model():
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
        "handwriting_recognizer_ru2.h5",
        custom_objects={"CTCLayer": CTCLayer},
        compile=False
    )

    _PREDICTION_MODEL = keras.Model(
        base_model.get_layer("image").output,
        base_model.get_layer("output").output
    )

def distortion_free_resize(image, img_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
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
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Не удалось загрузить изображение " + img_path)
    img = np.expand_dims(img, axis=-1)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = distortion_free_resize(img)
    img = img / 255.0
    return img

def decode_batch(pred):
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
    initialize_model()
    
    img = preprocess_image_cv2(img_path)
    img = tf.expand_dims(img, 0)
    preds = _PREDICTION_MODEL.predict(img, verbose=0)
    text = decode_batch(preds)[0]
    return text

if __name__ == "__main__":
    path = "14ru.jpg"
    result = predict_image(path)
    print("Распознанный текст:", result)