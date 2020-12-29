# -*- coding: utf-8 -*-
from keras import backend as K
K.set_image_dim_ordering('tf')
import keras
from keras.models import Model
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Input, Dense, Conv2D, Conv1D, Flatten, BatchNormalization, LeakyReLU, Dropout, \
    concatenate, GlobalAveragePooling2D, Activation, multiply, Permute, dot, Lambda
from keras.layers.core import Reshape
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
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
class_num = 16

def lambda_expand_dim(x):
    return K.expand_dims(x, axis=len(x.shape))


def spa_attention(x):
    VIS_conv1 = Conv3D(64, (1, 1, 200), padding='valid', strides=(1, 1, 1))(x)
    VIS_BN1 = BatchNormalization()(VIS_conv1)
    VIS_relu1 = PReLU()(VIS_BN1)
    VIS_SHAPE1 = Reshape((VIS_relu1._keras_shape[1] * VIS_relu1._keras_shape[2], VIS_relu1._keras_shape[4]))(
        VIS_relu1)

    VIS_conv2 = Conv3D(64, (1, 1, 200), padding='valid', strides=(1, 1, 1))(x)
    VIS_BN2 = BatchNormalization()(VIS_conv2)
    VIS_relu2 = PReLU()(VIS_BN2)
    VIS_SHAPE2 = Reshape((VIS_relu2._keras_shape[1] * VIS_relu2._keras_shape[2], VIS_relu2._keras_shape[4]))(
        VIS_relu2)
    trans_VIS_SHAPE2 = Permute((2, 1))(VIS_SHAPE2)

    VIS_conv3 = Conv3D(64, (1, 1, 200), padding='valid', strides=(1, 1, 1))(x)
    VIS_BN3 = BatchNormalization()(VIS_conv3)
    VIS_relu3 = PReLU()(VIS_BN3)
    VIS_SHAPE3 = Reshape((VIS_relu3._keras_shape[1] * VIS_relu3._keras_shape[2], VIS_relu3._keras_shape[4]))(
        VIS_relu3)

    VIS_mul1 = dot([VIS_SHAPE1, trans_VIS_SHAPE2], axes=(2, 1))

    VIS_sigmoid = Activation('softmax')(VIS_mul1)

    VIS_mul2 = dot([VIS_sigmoid, VIS_SHAPE3], axes=(2, 1))
    VIS_SHAPEall = Reshape((7, 7, 64, 1))(VIS_mul2)
    VIS_conv4 = Conv3D(200, (1, 1, 64), padding='valid', strides=(1))(VIS_SHAPEall)
    VIS_BN4 = BatchNormalization()(VIS_conv4)
    VIS_relu4 = PReLU()(VIS_BN4)
    VIS_conv4_shape = Reshape((VIS_relu4._keras_shape[1], VIS_relu4._keras_shape[2], VIS_relu4._keras_shape[4], 1))(
        VIS_relu4)
    VIS_conv5 = Conv3D(32, (1, 1, 1), padding='valid', strides=(1))(VIS_conv4_shape)
    VIS_BN5 = BatchNormalization()(VIS_conv5)
    VIS_relu5 = PReLU()(VIS_BN5)
    VIS_ADD = add([x, VIS_relu5])
    return  VIS_ADD

def spa_attention1(x):
    VIS_conv1 = Conv3D(64, (1, 1, 200), padding='valid', strides=(1, 1, 1))(x)
    VIS_BN1 = BatchNormalization()(VIS_conv1)
    VIS_relu1 = PReLU()(VIS_BN1)
    VIS_SHAPE1 = Reshape((VIS_relu1._keras_shape[1] * VIS_relu1._keras_shape[2], VIS_relu1._keras_shape[4]))(
        VIS_relu1)

    VIS_conv2 = Conv3D(64, (1, 1, 200), padding='valid', strides=(1, 1, 1))(x)
    VIS_BN2 = BatchNormalization()(VIS_conv2)
    VIS_relu2 = PReLU()(VIS_BN2)
    VIS_SHAPE2 = Reshape((VIS_relu2._keras_shape[1] * VIS_relu2._keras_shape[2], VIS_relu2._keras_shape[4]))(
        VIS_relu2)
    trans_VIS_SHAPE2 = Permute((2, 1))(VIS_SHAPE2)

    VIS_conv3 = Conv3D(64, (1, 1, 200), padding='valid', strides=(1, 1, 1))(x)
    VIS_BN3 = BatchNormalization()(VIS_conv3)
    VIS_relu3 = PReLU()(VIS_BN3)
    VIS_SHAPE3 = Reshape((VIS_relu3._keras_shape[1] * VIS_relu3._keras_shape[2], VIS_relu3._keras_shape[4]))(
        VIS_relu3)

    VIS_mul1 = dot([VIS_SHAPE1, trans_VIS_SHAPE2], axes=(2, 1))

    VIS_sigmoid = Activation('softmax')(VIS_mul1)

    VIS_mul2 = dot([VIS_sigmoid, VIS_SHAPE3], axes=(2, 1))
    VIS_SHAPEall = Reshape((7, 7, 64, 1))(VIS_mul2)
    VIS_conv4 = Conv3D(200, (1, 1, 64), padding='valid', strides=(1))(VIS_SHAPEall)
    VIS_BN4 = BatchNormalization()(VIS_conv4)
    VIS_relu4 = PReLU()(VIS_BN4)
    VIS_conv4_shape = Reshape((VIS_relu4._keras_shape[1], VIS_relu4._keras_shape[2], VIS_relu4._keras_shape[4], 1))(
        VIS_relu4)
    VIS_conv5 = Conv3D(32, (1, 1, 1), padding='valid', strides=(1))(VIS_conv4_shape)
    VIS_BN5 = BatchNormalization()(VIS_conv5)
    VIS_relu5 = PReLU()(VIS_BN5)
    VIS_ADD = add([x, VIS_relu5])
    return  VIS_ADD


def model():
    input_1 = Input(shape=(7,7,200))

    input_spe = Lambda(lambda_expand_dim)(input_1)

    conv_spe1 = Conv3D(filters=32, kernel_size=(1, 1, 7), padding='same', name='conv_spe1', data_format='channels_last')(input_spe)
    # conv_spe1 = Conv3D(32, (1,1,1), padding='same', strides=(1,1,1))(input_spe)
    bn_spe1 = BatchNormalization()(conv_spe1)
    relu_spe1 = PReLU()(bn_spe1)
    conv_spe2 = Conv3D(32,(1,1,7), padding='same')(relu_spe1)
    bn_spe2 = BatchNormalization()(conv_spe2)
    relu_spe2 = PReLU()(bn_spe2)

    VIS_ADD = spa_attention(relu_spe2)

    conv_spa1 = Conv3D(32,(3,3,200),padding='same')(VIS_ADD)
    bn_spa1 = BatchNormalization()(conv_spa1)
    relu_spa1 = PReLU()(bn_spa1)

    VIS_ADD1 = spa_attention1(relu_spa1)

    conv_spa2 = Conv3D(32, (3, 3, 200), padding='same')(VIS_ADD1)
    bn_spa2 = BatchNormalization()(conv_spa2)
    relu_spa2 = PReLU()(bn_spa2)
    VIS_ADD2 = spa_attention1(relu_spa2)
    fla = Flatten()(VIS_ADD2)

    fc1 = Dense(256, activation='relu')(fla)
    drop = Dropout(0.5)(fc1)

    output = Dense(class_num, activation='softmax')(drop)
    model = Model(input_1, output)

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model




