from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import network

tf.logging.set_verbosity(tf.logging.INFO)

#tf.enable_eager_execution() # TODO: check this


# def preprocess_data(data, image_count):
#     BATCH_SIZE = 32
#
#     # Setting a shuffle buffer size as large as the dataset ensures that the data is
#     # completely shuffled.
#     ds = data.shuffle(buffer_size=image_count)
#     ds = ds.repeat()
#     ds = ds.batch(BATCH_SIZE)
#     # `prefetch` lets the dataset fetch batches, in the background while the model is training.
#     ds = ds.prefetch(buffer_size=64)
#     return ds

def train_input_fn(features, labels):

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        return image_decoded, label

    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    dataset = dataset.map(_parse_function)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat()

    # Return the dataset.
    return dataset

labels_list = ["BomagCompactor", "CAT305EExcavator", "CATC5KBulldozer",
               "Hitachi50UExcavator", "JD50GExcavator1", "JD700JDozer"]

def load_files():

    fnames = []
    lbls = []
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.


    for dir_path, subdir_list, file_list in os.walk("./data"):
        for fname in file_list:
            if os.path.splitext(fname)[1] == ".jpg":
                full_file_name_with_path = os.path.join(dir_path, fname)
                fnames.append(full_file_name_with_path)
                lbls.append(labels_list.index(os.path.dirname(dir_path).split("./data/",1)[1]))

    # A vector of filenames.
    filenames = tf.constant(fnames)
    image_count = len(fnames)

    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant(lbls)

    #dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    return filenames, labels

def main():
    filenames, labels = load_files()
    #data = data.shuffle(1000).repeat().batch(100)
    print(filenames, labels)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=network.Network.build_network, model_dir="./tmp/audio_conv_model")


    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # est =

    # fn = tf.estimator.inputs.numpy_input_fn(x={"x": data}, y=data, batch_size=100, num_epochs=None, shuffle=True)
    # train_input_fn = fn

    mnist_classifier.train(
        steps=1000,
        input_fn=lambda: train_input_fn(filenames, labels)
    )

if __name__ == "__main__":
    main()

