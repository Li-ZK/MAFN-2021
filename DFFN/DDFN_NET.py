# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:11:02 2019

@author: admin
"""

import numpy as np

np.random.seed(1337)
from keras.layers import Conv3D, BatchNormalization, concatenate, Flatten, Dropout, Dense, AveragePooling3D
from keras.layers import Conv2D, AveragePooling2D, MaxPool3D, add, Activation
from keras.layers.advanced_activations import PReLU, LeakyReLU, ReLU
from keras.layers import Input
from keras.layers.core import Reshape
from keras.models import Model
from keras.optimizers import Adam

IN_input_shape = (25, 25, 3)
UP_input_shape = (23, 23, 5)
KSC_input_shape = (27, 27, 9)
HT_input_shape = (23,23,5)
class_num = 16


def Res_block(input_tensor, k_num, k_size):
    conv1 = Conv2D(k_num, (k_size, k_size), padding='same')(input_tensor)
    b1 = BatchNormalization()(conv1)
    r1 = ReLU()(b1)
    conv2 = Conv2D(k_num, (k_size, k_size), padding='same')(r1)
    b2 = BatchNormalization()(conv2)
    conv2 = add([input_tensor, b2])
    r2 = ReLU()(conv2)
    return r2


def DFFN():
    input_data = Input(shape=IN_input_shape)

    conv1 = Conv2D(16, (3, 3), padding='same')(input_data)
    b1 = BatchNormalization()(conv1)
    r1 = ReLU()(b1)

    low_fea = Res_block(r1, 16, 3)
    low_fea = Res_block(low_fea, 16, 3)
    low_fea = Res_block(low_fea, 16, 32)
    low_fea = Res_block(low_fea, 16, 32)
    low_fea = Res_block(low_fea, 16, 32)
    Low_Feature = Conv2D(64, (1, 1), padding='same')(low_fea)

    conv2 = Conv2D(32, (3, 3), padding='same')(low_fea)
    b2 = BatchNormalization()(conv2)
    r2 = ReLU()(b2)

    mid_fea = Res_block(r2, 32, 3)
    mid_fea = Res_block(mid_fea, 32, 3)
    mid_fea = Res_block(mid_fea, 32, 3)
    mid_fea = Res_block(mid_fea, 32, 3)
    mid_fea = Res_block(mid_fea, 32, 3)
    Mid_Feature = Conv2D(64, (1, 1), padding='same')(mid_fea)

    conv3 = Conv2D(64, (3, 3), padding='same')(mid_fea)
    b3 = BatchNormalization()(conv3)
    r3 = ReLU()(b3)

    high_fea = Res_block(r3, 64, 3)
    high_fea = Res_block(high_fea, 64, 3)
    high_fea = Res_block(high_fea, 64, 3)
    high_fea = Res_block(high_fea, 64, 3)
    high_fea = Res_block(high_fea, 64, 3)
    High_Feature = Conv2D(64, (1, 1), padding='same')(high_fea)

    fusion_fea = add([Low_Feature, Mid_Feature])
    fusion_fea = add([fusion_fea, High_Feature])
    p = AveragePooling2D((23, 23))(fusion_fea)
    fla = Flatten()(p)
    pred = Dense(class_num, activation='softmax')(fla)

    model = Model(input_data, pred)
    adam = Adam(lr=0.0003)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model


