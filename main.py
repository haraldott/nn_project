from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import sys
import random
import pathlib
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping

tf.logging.set_verbosity(tf.logging.INFO)
tf.enable_eager_execution()

# HYPERPARAMETERS
BATCH_SIZE = 6

working_dir = str(sys.argv[1])
data_root = pathlib.Path(working_dir + "/data")
AUTOTUNE = tf.data.experimental.AUTOTUNE


def build_network():

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(140, 140, 3), filters=80, kernel_size=[11, 11], activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[4, 3], strides=[4, 4]),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(filters=80, kernel_size=[10, 10], activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 3], strides=[2, 2]),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(filters=80, kernel_size=[9, 9], activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 3], strides=[2, 2]),
        tf.keras.layers.Dense(units=2000, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=2000),
        tf.keras.layers.Dense(units=6, activation=tf.nn.softmax)
    ])
    return model

earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

def load_files():

    # get all labels
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

    # assign index to each label
    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    # get all image paths
    all_image_paths = list(data_root.glob('*/spectrogram/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    # create a list of every file and its label index
    all_image_labels = [label_to_index[pathlib.Path(path).parent.parent.name]
                        for path in all_image_paths]

    def _preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [140, 140])
        #image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        return _preprocess_image(image)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int32))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds, len(all_image_paths)

def main():

    ds, len_images = load_files()
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    model = build_network()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=["accuracy"])
    print(model.summary())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)

    steps_per_epoch = int(tf.ceil(len_images / BATCH_SIZE).numpy())

    model.fit(ds, epochs=2, steps_per_epoch=steps_per_epoch)
    scores = model.evaluate(ds, verbose=0, steps=BATCH_SIZE)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open(working_dir + "model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(working_dir + "model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    main()

