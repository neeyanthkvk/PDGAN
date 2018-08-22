#https://github.com/khushhallchandra/keras-3dgan/blob/master/src/models.py
from __future__ import print_function, division

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import numpy as np
import os


def generator(phase_train=True, params={'latent_dim':200, 'strides':(2,2,2), 'kernel_size':(4,4,4)}):
    l_dim = params['latent_dim']
    strides = params['strides']
    kernel_size = params['kernel_size']

    inputs = Input(shape=(1,1,1,l_dim))

    group1 = Deconv3D(512, kernel_size, kernel_initializer='glorot_normal', bias_initlizer='zeros')(inputs)
    group1 = BatchNormalization()(group1, training=phase_train)
    group1 = Activation(activation='relu')(group1)

    group2 = Deconv3D(256, kernel_size, strides = strides, kernel_initializer='glorot_normal', bias_initlizer='zeros')(group1)
    group2 = BatchNormalization()(group2, training=phase_train)
    group2 = Activation(activation='relu')(group2)

    group3 = Deconv3D(128, kernel_size, strides = strides, kernel_initializer='glorot_normal', bias_initlizer='zeros')(group2)
    group3 = BatchNormalization()(group3, training=phase_train)
    group3 = Activation(activation='relu')(group3)

    group4 = Deconv3D(32, kernel_size, strides = strides, kernel_initializer='glorot_normal', bias_initlizer='zeros')(group3)
    group4 = BatchNormalization()(group4, training=phase_train)
    group4 = Activation(activation='relu')(group4)

    group5 = Deconv3D(16, kernel_size, strides = strides, kernel_initializer='glorot_normal', bias_initlizer='zeros')(group4)
    group5 = BatchNormalization()(group5, training=phase_train)
    group5 = Activation(activation='relu')(group5)

    group6 = Deconv3D(1, kernel_size, strides = strides, kernel_initializer='glorot_normal', bias_initlizer='zeros')(group5)
    group6 = BatchNormalization()(group6, training=phase_train)
    group6 = Activation(activation='relu')(group6)

    geners = Model(inputs, group6)
    geners.summary()
    return geners

def discriminator(phase_train = True, params={'shape':64, 'strides':(2,2,2), 'kernel_size':(4,4,4), 'leak_value':0.2}):
    shape = params['shape256']
    strides = params['strides']
    kernel_size = params['kernel_size']
    leak_value = params['leak_value']

    inputs = Input(shape=shape)

    group1 = Conv3D(64, kernel_size = kernel_size, strides = strides, kernel_initializer = 'glorot_normal', bias_initlizer = 'zeros')(inputs)
    group1 = BatchNormalization()(group1, training=phase_train)
    group1 = LeakyReLU(leak_value)(group1)

    group2 = Conv3D(128, kernel_size = kernel_size, strides = strides, kernel_initializer = 'glorot_normal', bias_initlizer = 'zeros')(group1)
    group2 = BatchNormalization()(group2, training=phase_train)
    group2 = LeakyReLU(leak_value)(group2)

    group3 = Conv3D(256, kernel_size = kernel_size, strides = strides, kernel_initializer = 'glorot_normal', bias_initlizer = 'zeros')(group2)
    group3 = BatchNormalization()(group3, training=phase_train)
    group3 = LeakyReLU(leak_value)(group3)

    group4 = Conv3D(256, kernel_size = kernel_size, strides = strides, kernel_initializer = 'glorot_normal', bias_initlizer = 'zeros')(group3)
    group4 = BatchNormalization()(group4, training=phase_train)
    group4 = LeakyReLU(leak_value)(group4)

    discrims = Model(inputs, group4)
    discrims.summary()
    return discrims

if __name__ == "__main__":
    shap = (176, 256, 240, 1)
    gen = generator(True, {'latent_dim': 1000, 'strides':(2,2,2), 'kernel_size':(5,5,5)})
    disc = discriminator(params={'shape':shap, 'strides':(2,2,2), 'kernel_size':(5,5,5), 'leak_value':0.25})

    '''
    Hyper-parameters
    '''
    epchs = 100
    batch_size = 1
    g_lr       = 0.008
    d_lr       = 0.000001
    beta       = 0.5
    z_size     = 200
    cube_len   = 64
    obj_ratio  = 0.5

    train_dir = ""
    model_dir = ""

    model = Sequential()
    model.add(gen)
    disc.trainable = False
    model.add(disc)

    g_optimizer = Adam(lr = g_lr, beta_1 = beta)
    d_optimizer = Adam(lr = d_lr, beta_1 = beta)

    gen.compile(loss = 'binary_crossentropy', optimizer='SGD')
    model.compile(loss = 'binary_crossentropy', optimizer = g_optimizer)
    disc.trainable = True
    disc.compiler(loss = 'binary_crossentropy', optimizer = d_optimizer)

    
