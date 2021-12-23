# UNet with Dilation Implementation with Keras
# Inspired by https://github.com/saedrna/DA-U-Net/blob/a260155448abbcc3018bdd9027cfb72af3cca2ae/code/model.py
import os
import numpy as np

from tensorflow.keras.optimizers import *
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import Dropout, BatchNormalization, Activation

from keras.layers import concatenate, add
from keras.losses import binary_crossentropy
from metric_loss import f1


# Convolution Block for UNet
def conv_block(inputs, n_filter, k_size=3, activation='relu'):
    x = Conv2D(n_filter, kernel_size=k_size, padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(n_filter, kernel_size=k_size, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x


# Dilation Bottom Block
def dilation_bottom(x, n_filter, k_size=3, dilation_length=6, activation='relu'):
    d_layer_list = []

    for d_step in range(dilation_length):
        x = Conv2D(n_filter, kernel_size=k_size, activation=activation, dilation_rate=2**d_step, padding='same')(x)
        d_layer_list.append(x)

    d_layers = add(d_layer_list)
    return d_layers


# Dropout Block
def dropout(x, d_out=True, d_rate=0.5):
    if d_out is True:
        x = Dropout(d_rate)(x)
    return x


# Standard UNet Model
def dilated_unet(size=(None, None, 3), n_filter=64, activation='relu', d_out=True, d_rate=0.5,
         loss=binary_crossentropy, lr_rate=1e-4, model_path=None):
    """
    Arguments:
        size: size of input images
        n_filter: number of filters of the 1st layer
        activation: activation function to use
        d_out: flag of dropout layer
        d_rate: dropout rate
        loss: loss function to use (default: binary_crossentropy)
        lr_rate: learning rate of Adam Optimizer (default: 1e-4)
        model_path: load pretrained weights if exists

    Return:
        UNet model to train
    """

    inputs = Input(shape=size)

    # Convolution
    conv1 = conv_block(inputs, n_filter, k_size=3, activation=activation)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, n_filter * 2, k_size=3, activation=activation)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, n_filter * 4, k_size=3, activation=activation)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom Dilated Convolution
    bottom = dilation_bottom(pool3, n_filter * 8, k_size=3, dilation_length=6, activation=activation)

    # Deconvolution
    dconv1 = Conv2DTranspose(n_filter * 4, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(bottom)
    merge1 = concatenate([conv3, dconv1], axis=3)
    merge1 = dropout(merge1, d_out, d_rate)
    conv4 = conv_block(merge1, n_filter * 4, k_size=3, activation=activation)

    dconv2 = Conv2DTranspose(n_filter * 2, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv4)
    merge2 = concatenate([conv2, dconv2], axis=3)
    merge2 = dropout(merge2, d_out, d_rate)
    conv5 = conv_block(merge2, n_filter * 2, k_size=3, activation=activation)

    dconv3 = Conv2DTranspose(n_filter, kernel_size=2, strides=2, kernel_initializer="he_normal", padding='same')(conv5)
    merge3 = concatenate([conv1, dconv3], axis=3)
    merge3 = dropout(merge3, d_out, d_rate)
    conv6 = conv_block(merge3, n_filter, k_size=3, activation=activation)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv6)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=lr_rate), loss=loss, metrics=[f1, 'accuracy'])
    model.summary()

    # Load previous model weights if exist
    if model_path:
        model.load_weights(filepath=model_path)

    return model
