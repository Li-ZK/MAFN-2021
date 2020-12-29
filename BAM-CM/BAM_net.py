from keras import backend as K
K.set_image_dim_ordering('th')
import keras
from keras.models import Model
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Input, Dense, Conv2D, Conv1D, Flatten, BatchNormalization, LeakyReLU, Dropout, \
    concatenate, GlobalAveragePooling2D, Activation, multiply, Permute, dot
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
class_num = 15

def model():

    input_1 = Input(shape=(15,15,144))
    CAB_conv1 = Conv2D(16, (3, 3), padding='same', strides=(1, 1),
                       kernel_initializer='he_normal',
                       )(input_1)
    CAB_bn1 = BatchNormalization()(CAB_conv1)
    CAB_relu1 = PReLU()(CAB_bn1)
    CAB_avg_pool1 = AveragePooling2D()(CAB_relu1)
    # --------------------------------------------------------------------------------------------------
    CAB_conv2 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                       kernel_initializer='he_normal',

                       )(CAB_avg_pool1)
    CAB_bn2 = BatchNormalization()(CAB_conv2)
    CAB_relu2 = PReLU()(CAB_bn2)
    CAB_conv3 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                       kernel_initializer='he_normal',

                       )(CAB_relu2)
    CAB_bn3 = BatchNormalization()(CAB_conv3)
    CAB_relu3 = PReLU()(CAB_bn3)
    CAB_avg_pool2 = AveragePooling2D()(CAB_relu3)
    # ==================================================================================================================
    CAB_conv4 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                       kernel_initializer='he_normal',

                       )(CAB_avg_pool2)
    CAB_bn4 = BatchNormalization()(CAB_conv4)
    CAB_relu4 = PReLU()(CAB_bn4)
    CAB_conv5 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                       kernel_initializer='he_normal',

                       )(CAB_relu4)
    CAB_bn5 = BatchNormalization()(CAB_conv5)
    CAB_relu5 = PReLU()(CAB_bn5)
    CAB_global_pool = GlobalAveragePooling2D()(CAB_relu5)
    #===================================================================================================================
    CAB_reshape = Reshape((1, CAB_global_pool._keras_shape[1]))(CAB_global_pool)

    CAB_conv6 = Conv1D(48, (32), padding='same', strides=(1))(CAB_reshape)
    CAB_bn6 = BatchNormalization()(CAB_conv6)
    CAB_relu6 = PReLU()(CAB_bn6)

    CAB_conv7 = Conv1D(144, (48), padding='same', strides=(1))(CAB_relu6)
    CAB_bn7 = BatchNormalization()(CAB_conv7)
    CAB_relu7 = PReLU()(CAB_bn7)
    CAB_sigmoid = Activation('sigmoid')(CAB_relu7)
    # ==================================================================================================================
    CAB_mul = multiply([input_1, CAB_sigmoid])

    conv1 = Conv2D(16,(3,3),padding='same')(CAB_mul)
    bn1 = BatchNormalization()(conv1)
    relu1 = PReLU()(bn1)
    pool1 = MaxPooling2D((2,2))(relu1)

    conv2 = Conv2D(32, (3, 3),padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    relu2 = PReLU()(bn2)
    pool2 = MaxPooling2D((2, 2))(relu2)

    conv3 = Conv2D(32, (3, 3),padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    relu3 = PReLU()(bn3)
    fla = Flatten()(relu3)

    output = Dense(class_num, activation='softmax')(fla)
    model = Model(input_1, output)
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

