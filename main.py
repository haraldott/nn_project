from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import network
import sys
import pathlib
import random

tf.logging.set_verbosity(tf.logging.INFO)
tf.enable_eager_execution()

# HYPERPARAMETERS
BATCH_SIZE = 32


data_root = pathlib.Path(sys.argv[1])
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
        image = tf.read_file(str(path))
        return _preprocess_image(image)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    print(image_ds)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    # image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_ds, label_ds, len(all_image_paths)

def main():

    image_ds, label_ds, len_images = load_files()
    # ds = ds.repeat()
    # ds = ds.batch(BATCH_SIZE)
    # ds = ds.prefetch(buffer_size=AUTOTUNE)

    # def change_range(image, label):
    #     return 2 * image - 1, label
    #
    # keras_ds = ds.map(change_range)

    print("len_images ", len_images)
    model = network.build_network()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=["accuracy"])
    print(model.summary())

    steps_per_epoch = int(tf.ceil(len_images / BATCH_SIZE).numpy())

    #serialise
    ds = image_ds.map(tf.serialize_tensor)
    if not os.path.isfile('images.tfrec'):
        tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
        tfrec.write(ds)


    RESTORE_TYPE = image_ds.output_types
    RESTORE_SHAPE = image_ds.output_shapes

    ds = tf.data.TFRecordDataset('images.tfrec')

    def parse(x):
        result = tf.parse_tensor(x, out_type=RESTORE_TYPE)
        result = tf.reshape(result, RESTORE_SHAPE)
        return result

    ds = ds.map(parse, num_parallel_calls=AUTOTUNE)

    ds = tf.data.Dataset.zip((ds, label_ds))
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=len_images))
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


    model.fit(ds, epochs=2, steps_per_epoch=steps_per_epoch)
    scores = model.evaluate(ds, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    main()

