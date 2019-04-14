from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import network
import sys
import pathlib
from keras.backend.tensorflow_backend import set_session

tf.logging.set_verbosity(tf.logging.INFO)
tf.enable_eager_execution()

# HYPERPARAMETERS
BATCH_SIZE = 32

working_dir = str(sys.argv[1])
data_root = pathlib.Path(working_dir + "/data")
AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_files():

    # get all labels
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print(label_names)

    # assign index to each label
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print(label_to_index)

    # get all image paths
    all_image_paths = list(data_root.glob('*/spectrogram/*'))
    all_image_paths = [str(path) for path in all_image_paths]

    # create a list of every file and its label index
    all_image_labels = [label_to_index[pathlib.Path(path).parent.parent.name]
                        for path in all_image_paths]

    def _preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [192, 192])
        image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        return _preprocess_image(image)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    print(image_ds)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds, len(all_image_paths)

def main():

    ds, len_images = load_files()
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    model = network.build_network()
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

    print(len(model.trainable_variables))
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

