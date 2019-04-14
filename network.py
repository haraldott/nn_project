from __future__ import absolute_import, division, print_function

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping

tf.logging.set_verbosity(tf.logging.INFO)
tf.enable_eager_execution()


def build_network():

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(192, 192, 3), filters=80, kernel_size=[57, 6], activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[4, 3], strides=[1, 3]),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(filters=80, kernel_size=[1, 3], activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[1, 3], strides=[1, 3]),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=5000, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=5000),
        tf.keras.layers.Dense(units=6, activation=tf.nn.softmax)
    ])
    return model

    # model.add()
    #
    # model.add()
    #
    # model.add()
    #
    # model.add()
    #
    # model.add()
    #
    # model.add()
    #
    # model.add()
    #
    # model.add()
    #
    # model.add()
    #
    # model.add()
    #
    # model.add()
    #
    # return model

earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

#class Network(object):




        #input_layer = tf.reshape(features["x"], [-1, 921600])

        #conv_1 = tf.layers.conv2d(inputs=input_layer, filters=80, kernel_size=[57,6], activation=tf.nn.relu)

        #pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[4,3], strides=[1,3])

        #dropout_1 = tf.layers.dropout(inputs=pool_1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        #conv_2 = tf.layers.conv2d(inputs=dropout_1, filters=80, kernel_size=[1, 3], activation=tf.nn.relu)

        #pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[1,3], strides=[1,3])

        #dropout_2 = tf.layers.dropout(inputs=pool_2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        #dense_layer_1 = tf.layers.dense(inputs=dropout_2, num_outputs=5000, activation=tf.nn.relu)

        #dropout_3 = tf.layers.dropout(inputs=dense_layer_1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

        #dense_layer_2 = tf.layers.dense(inputs=dropout_3, num_outputs=5000)

        #logits = tf.layers.dense(inputs=dense_layer_2, units=6)

        # predictions = {
        #     "classes": tf.argmax(input = logits, axis=1),
        #     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        # }
        #
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        #
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        #
        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        #     train_op = optimizer.minimize(
        #         loss=loss,
        #         global_step=tf.train.get_global_step())
        #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        #
        # eval_metric_ops = {
        #     "accuracy": tf.metrics.accuracy(
        #         labels=labels, predictions=predictions["classes"])
        # }
        #
        # return tf.estimator.EstimatorSpec(
        #     mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        #

