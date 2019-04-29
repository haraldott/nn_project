from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.utils import np_utils

frames = 41
bands = 60
feature_size = bands * frames #60x41
num_labels = 6
num_channels = 2


def build_model():
    model = Sequential()
    # input: 60x41 data frames with 2 channels => (60,41,2) tensors

    # filters of size 1x1
    f_size = 1

    # first layer has 48 convolution filters
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same',
                            input_shape=(bands, frames, num_channels)))
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # next layer has 96 convolution filters
    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # flatten output into a single dimension
    # Keras will do shape inference automatically
    model.add(Flatten())

    # then a fully connected NN layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # finally, an output layer with one node per class
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    # use the Adam optimiser
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    return model
