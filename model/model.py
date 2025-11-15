import keras
from keras.layers import StringLookup
from keras import ops
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from keras import layers, backend as K
import pandas as pd

np.random.seed(42)
keras.utils.set_random_seed(42)
import pandas as pd

# Load data
train_labels_df = pd.read_csv('./data_ru/train.tsv', sep='\t', header=None, names=['image_name', 'label'])
test_labels_df = pd.read_csv('./data_ru/test.tsv', sep='\t', header=None, names=['image_name', 'label'])

# Remove 'err' and labels with 2+ words
train_labels_df = train_labels_df[
    (train_labels_df['label'] != 'err') &
    (train_labels_df['label'].str.split().str.len() == 1)
]
train_labels_df['combined'] = train_labels_df['image_name'] + " " + train_labels_df['label']

test_labels_df = test_labels_df[
    (test_labels_df['label'] != 'err') &
    (test_labels_df['label'].str.split().str.len() == 1)
]
test_labels_df['combined'] = test_labels_df['image_name'] + " " + test_labels_df['label']

# Base path (if needed elsewhere)
base_path = "data_ru"

# Create list of "image_name label" lines
words_list = train_labels_df['combined'].tolist() + test_labels_df['combined'].tolist()


np.random.shuffle(words_list)

split_idx = int(0.9 * len(words_list))
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]


val_split_idx = int(0.5 * len(test_samples))
validation_samples = test_samples[:val_split_idx]
test_samples = test_samples[val_split_idx:]

assert len(words_list) == len(train_samples) + len(validation_samples) + len(
    test_samples
)

print(f"Total training samples: {len(train_samples)}")
print(f"Total validation samples: {len(validation_samples)}")
print(f"Total test samples: {len(test_samples)}")


base_image_path = os.path.join(base_path)

def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []

    for i, file_line in enumerate(samples):
        # Приводим к строке
        file_line = str(file_line).strip()
        if not file_line:  # пропускаем пустые строки
            continue

        image_name = file_line.split()[0]  # Берем только имя файла
        folder = "test" if image_name.startswith("test") else "train"
        img_path = os.path.join(base_image_path, folder, image_name)

        if i % 1000 == 0:
            print(f"Processing line {i}: {image_name}")

        if os.path.exists(img_path):
            if os.path.getsize(img_path) > 0:
                paths.append(img_path)
                corrected_samples.append(file_line)
            else:
                print(f"Skipped empty file: {img_path}")
        else:
            print(f"File not found: {img_path}")

    return paths, corrected_samples



train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)

# Find maximum length and the size of the vocabulary in the training data.
train_labels_cleaned = []
characters = set()
max_len = 0
#
for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)

    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)
#
characters = sorted(list(characters))
#
print("Maximum length: ", max_len)
print("Vocab size: ", len(characters))
print("Vocab value: ", characters)
#
# # Check some label samples.
print(train_labels_cleaned[:10])
#
def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels


validation_labels_cleaned = clean_labels(validation_labels)
test_labels_cleaned = clean_labels(test_labels)
#
#
AUTOTUNE = tf.data.AUTOTUNE
#
# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
#
#
def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - ops.shape(image)[0]
    pad_width = w - ops.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

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
#
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32
#
#
def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = ops.cast(image, tf.float32) / 255.0
    return image
#
#
def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = ops.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label
#
#
def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}
#
#
def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

train_ds = prepare_dataset(train_img_paths, train_labels_cleaned)
validation_ds = prepare_dataset(validation_img_paths, validation_labels_cleaned)
test_ds = prepare_dataset(test_img_paths, test_labels_cleaned)
#
for data in train_ds.take(1):
    images, labels = data["image"], data["label"]

    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    for i in range(16):
        img = images[i]
        img = tf.image.flip_left_right(img)
        img = ops.transpose(img, (1, 0, 2))
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        # Gather indices where label!= padding_token.
        label = labels[i]
        indices = tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))
        # Convert to string.
        label = tf.strings.reduce_join(num_to_char(indices))
        label = label.numpy().decode("utf-8")

        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
#
#
plt.show()

# Значок паддинга — убедитесь, что это то же значение, что вы используете при vectorize_label
padding_token = 99

class CTCLayer(keras.layers.Layer):
    def __init__(self, padding_token, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost
        self.padding_token = padding_token

    def call(self, y_true, y_pred):
        # y_true: (batch, max_label_len)  — содержит padded labels
        # y_pred: (batch, time_steps, vocab_size)
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        time_steps = ops.cast(ops.shape(y_pred)[1], dtype="int64")

        # input_length: одинаков для всех в батче — число временных шагов модели
        input_length = time_steps * ops.ones(shape=(batch_len, 1), dtype="int64")

        # label_length: реальная длина каждой метки (без паддинга)
        # предполагается, что padding_token однозначно обозначает паддинг
        # получаем булевую маску ненулевых/непаддинг позиций и суммируем по оси 1
        mask = tf.math.not_equal(y_true, self.padding_token)
        label_length = ops.cast(tf.reduce_sum(ops.cast(mask, dtype="int32"), axis=1), dtype="int64")
        # сделать shape=(batch,1)
        label_length = ops.reshape(label_length, (batch_len, 1))

        # tf.keras.backend.ctc_batch_cost ожидает y_true (dense), y_pred, input_length, label_length
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


#
#
def build_model():
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # Улучшенная CNN часть с batch normalization и residual connections
    x = keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same", name="Conv1"
    )(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # Второй блок с увеличенным количеством фильтров
    x = keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="Conv2"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(0.2)(x)

    # Третий сверточный блок
    x = keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="Conv3"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    # Дополнительный блок для извлечения более сложных features
    x = keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="Conv4"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((1, 2), name="pool3")(x)  # Только по высоте

    new_shape = ((image_width // 4), (image_height // 8) * 128)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    
    # Улучшенная Dense часть
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(64, activation="relu", name="dense2")(x)
    x = keras.layers.Dropout(0.2)(x)

    # Улучшенная RNN часть с GRU/LSTM
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(x)

    # Финальный слой
    x = keras.layers.Dense(
        len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="output"
    )(x)

    output = CTCLayer(padding_token=padding_token, name="ctc_loss")(labels, x)
    
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="improved_handwriting_recognizer"
    )

    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt)
    
    return model


# Get the model.
model = build_model()
model.summary()
#
validation_images = []
validation_labels = []

for batch in validation_ds:
    validation_images.append(batch["image"])
    validation_labels.append(batch["label"])

def calculate_edit_distance(labels, predictions):
    # Get a single batch and convert its labels to sparse tensors.
    saprse_labels = ops.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded = keras.ops.nn.ctc_decode(
        predictions, sequence_lengths=input_len
    )[0][0][:, :max_len]
    sparse_predictions = ops.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, saprse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(validation_images)):
            labels = validation_labels[i]
            predictions = self.prediction_model.predict(validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )

epochs = 100  # To get good results this should be at least 50.

model = build_model()
prediction_model = keras.models.Model(
    model.get_layer(name="image").output, model.get_layer(name="dense2").output
)
edit_distance_callback = EditDistanceCallback(prediction_model)

steps_per_epoch = len(train_ds)

periodic_checkpoint = ModelCheckpoint(
    filepath="checkpoints/model_epoch_{epoch:02d}.h5",
    save_best_only=False,
    save_weights_only=False,
    save_freq=5 * steps_per_epoch,  # сохранять каждые 5 эпох
    verbose=1,

)

checkpoint = ModelCheckpoint(
    filepath="checkpoint_best.h5",  # путь, куда сохранять модель
    monitor="val_loss",             # мониторим валидационный лосс
    save_best_only=True,            # сохраняем только, если стало лучше
    save_weights_only=False,        # сохранить всю модель
    verbose=1                       # печатать лог при сохранении
)
# Train the model.
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=[edit_distance_callback, checkpoint, periodic_checkpoint],
)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.ops.nn.ctc_decode(pred, sequence_lengths=input_len)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = (
            tf.strings.reduce_join(num_to_char(res))
            .numpy()
            .decode("utf-8")
            .replace("[UNK]", "")
        )
        output_text.append(res)
    return output_text


#  Let's check results on some test samples.
for batch in test_ds.take(1):
    batch_images = batch["image"]
    _, ax = plt.subplots(4, 4, figsize=(15, 8))

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    for i in range(16):
        img = batch_images[i]
        img = tf.image.flip_left_right(img)
        img = ops.transpose(img, (1, 0, 2))
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")

plt.show()

model.save("handwriting_recognizer_ru2.h5")