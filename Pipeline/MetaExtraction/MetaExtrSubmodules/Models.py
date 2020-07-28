from keras.layers import Conv1D
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.layers import GRU
from keras.layers import GaussianDropout
from keras.layers import Layer
from keras.layers.core import *


def build_RNN_model(num_of_classes=13, embed_dim=128):
    model = Sequential()
    model.add(Layer(input_shape=(None, embed_dim,)))
    model.add(Bidirectional(GRU(64, recurrent_dropout=0.2, return_sequences=True)))
    # model.add(GaussianDropout(0.05))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_of_classes, activation="softmax"))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def build_CNN_model(seq_len=31, num_of_classes=13, embed_dim=128):
    model = Sequential()
    model.add(Layer(input_shape=(seq_len, embed_dim,)))
    model.add(Conv1D(64, kernel_size=3))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_of_classes, activation="softmax"))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
