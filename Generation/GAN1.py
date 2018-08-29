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
from PIL import Image
import argparse

'''
Hyper-parameters
'''
epchs = 100
g_lr       = 0.008
d_lr       = 0.000001
beta       = 0.5
z_size     = 200
cube_len   = 64
obj_ratio  = 0.5
shap = (176, 256, 240, 1)

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.set_defaults(train=False)
    args = parser.parse_args()
    return args

def train(b_size, epchs):
    (x_train, y_train), (x_test, y_test) = (1,1), (1,1)
    x_train = (x_train.astype(np.float32) - 127.5)/127.5 #Scales from -1 to 1
    x_train = x_train[:,:,:,:,None]
    x_test = x_test[:,:,:,:,None]

    gen = generator(True, {'latent_dim': 1000, 'strides':(2,2,2), 'kernel_size':(5,5,5)})
    disc = discriminator(params={'shape':shap, 'strides':(2,2,2), 'kernel_size':(5,5,5), 'leak_value':0.25})

    finModel = Sequential()
    finModel.add(gen)
    disc.trainable = False
    finModel.add(disc)

    g_optimizer = Adam(lr = g_lr, beta_1 = beta)
    d_optimizer = Adam(lr = d_lr, beta_1 = beta)

    gen.compile(loss = 'binary_crossentropy', optimizer='SGD')
    model.compile(loss = 'binary_crossentropy', optimizer = g_optimizer)
    disc.trainable = True
    disc.compiler(loss = 'binary_crossentropy', optimizer = d_optimizer)

    for epoch in range(epchs):
        num_steps = int(X_train.shape[0]/b_size)
        for index in range(num_steps):
            noise = np.random.uniform(-1, 1, size=(b_size, 1000))
            image_batch = x_train[index*b_size:(index+1)*b_size]
            gen_img = gen.predict(noise, verbose = 0)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * b_size + [0] * b_size
            d_l = disc.train_on_batch(X, y)
            noise = np.random.uniform(-1, 1, size=(b_size, 1000))
            disc.trainable = False
            g_loss = model.train_on_batch(noise, [1] * BATCH_SIZE)
            disc.trainable = True
            if index % 10 == 9:
                gen.save_weights('generator', True)
                disc.save_weights('discriminator', True)

def gen(b_size, epchs):
    gen = generator(True, {'latent_dim': 1000, 'strides':(2,2,2), 'kernel_size':(5,5,5)})
    gen.compile(loss='binary_crossentropy', optimizer="SGD")
    gen.load_weights('generator')
    noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
    generated_images = gen.predict(noise, verbose=1)
    image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(b_size = args.batch_size, epchs = args.epoch)
    elif args.mode == "generate":
        gen(b_size = args.batch_size, epchs = args.epoch)
