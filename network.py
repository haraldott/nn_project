from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

frames = 41
bands = 60
feature_size = bands * frames #60x41
num_labels = 6
num_channels = 2


def build_model():
    model = Sequential()
    f_size = 1
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same',
                            input_shape=(bands, frames, num_channels)))
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    return model
