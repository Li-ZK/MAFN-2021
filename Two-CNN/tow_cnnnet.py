from keras import backend as K
K.set_image_dim_ordering('th')
import keras
from keras.models import Model
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Input, Dense, Conv2D, Conv1D, Flatten, BatchNormalization, LeakyReLU, Dropout, \
    concatenate, GlobalAveragePooling2D, Activation, multiply, Permute, dot
from keras.layers.core import Reshape
from keras.layers.merge import add, concatenate
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    AveragePooling2D,
    MaxPooling1D,
    Conv1D,
    Conv2D,
    Conv3D,
    MaxPooling2D,
    MaxPooling3D
)

from keras.optimizers import SGD, Adam
from keras.regularizers import l2

nb_classes = 15

num_filters_spe = 16
num_filters_spa = 16
r = 3


def model():
    input_1 = Input(shape=(1,1,176))
    input_2 = Input(shape=(21,21,1))
    input_spe =Reshape((input_1._keras_shape[3],input_1._keras_shape[1]*input_1._keras_shape[2]))(input_1)
    conv_spe1 = Conv1D(20, 16, strides=1, padding='same')(input_spe)
    conv_spe2 = Conv1D(20, 16, strides=1, padding='same')(conv_spe1)
    pool_spe = MaxPooling1D(2)(conv_spe2)
    flatten1 = Flatten()(pool_spe)

    input_spa = Reshape((input_2._keras_shape[3], input_2._keras_shape[2] , input_2._keras_shape[1]))(input_2)
    conv_spa1 = Conv2D(30,(3,3), strides=(1,1), padding='same')(input_spa)
    conv_spa2 = Conv2D(30,(3,3), strides=(1,1), padding='same')(conv_spa1)

    pool_spa = MaxPooling2D(pool_size=(2, 2),data_format="channels_first")(conv_spa2)
    flatten2 = Flatten()(pool_spa)

    addition = concatenate([flatten1, flatten2])

    fc1 = Dense(400, activation='relu')(addition)
    fc2 = Dense(400,  activation='relu')(fc1)
    output = Dense(nb_classes, activation='softmax')(fc2)
    model = Model(inputs=[input_1, input_2], outputs=output)
    sgd = keras.optimizers.sgd(lr=0.0001, momentum=0.9)
    # adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model






