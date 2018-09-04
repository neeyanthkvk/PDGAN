#https://github.com/khushhallchandra/keras-3dgan/blob/master/src/models.py
#https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
from __future__ import print_function, division

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Flatten, Dense
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
shap = (176, 32, 30, 1)

def generator(phase_train=True, params={'latent_dim':200, 'strides':(2,2,2)}):
    l_dim = params['latent_dim']
    strides = params['strides']

    inputs = Input(shape=(l_dim,))
    inputss = Reshape((22, 4, 1, 1), input_shape=(l_dim,))(inputs)
    group1 = Deconv3D(512, (2,2,2), strides = (2,2,2), kernel_initializer='glorot_normal', bias_initializer='zeros')(inputss)
    group1 = BatchNormalization()(group1, training=phase_train)
    group1 = Activation(activation='relu')(group1)

    group2 = Deconv3D(64, (2,2,2), strides = (2,2,3), kernel_initializer='glorot_normal', bias_initializer='zeros')(group1)
    group2 = BatchNormalization()(group2, training=phase_train)
    group2 = Activation(activation='relu')(group2)

    group3 = Deconv3D(1, (2,2,2), strides = (2,2,5), kernel_initializer='glorot_normal', bias_initializer='zeros')(group2)
    group3 = BatchNormalization()(group3, training=phase_train)
    group3 = Activation(activation='relu')(group3)

    geners = Model(inputs, group3)
    geners.summary()
    return geners

def discriminator(phase_train = True, params={'shape':shap, 'strides':(2,2,2), 'kernel_size':(4,4,4), 'leak_value':0.2}):
    shape = params['shape']
    strides = params['strides']
    kernel_size = params['kernel_size']
    leak_value = params['leak_value']

    inputs = Input(shape=shape)

    group1 = Conv3D(64, kernel_size = kernel_size, strides = strides, kernel_initializer = 'glorot_normal', bias_initializer = 'zeros')(inputs)
    group1 = BatchNormalization()(group1, training=phase_train)
    group1 = LeakyReLU(leak_value)(group1)

    group2 = Conv3D(128, kernel_size = kernel_size, strides = strides, kernel_initializer = 'glorot_normal', bias_initializer = 'zeros')(group1)
    group2 = BatchNormalization()(group2, training=phase_train)
    group2 = LeakyReLU(leak_value)(group2)

    group3 = Conv3D(256, kernel_size = kernel_size, strides = strides, kernel_initializer = 'glorot_normal', bias_initializer = 'zeros')(group2)
    group3 = BatchNormalization()(group3, training=phase_train)
    group3 = LeakyReLU(leak_value)(group3)

    group4 = Conv3D(256, kernel_size = kernel_size, strides = strides, kernel_initializer = 'glorot_normal', bias_initializer = 'zeros')(group3)
    group4 = BatchNormalization()(group4, training=phase_train)
    group4 = LeakyReLU(leak_value)(group4)

    group5 = Flatten()(group4)
    group5 = Dense(64, activation = 'relu')(group5)
    group5 = Dense(1, activation = 'sigmoid')(group5)

    discrims = Model(inputs, group5)
    discrims.summary()
    return discrims

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--b_size", type=int, default=4)
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img", type=int, default=50)
    parser.set_defaults(train=False)
    args = parser.parse_args()
    return args

def train(b_size, epchs):
    PD = np.load("/data/PD.npy")
    Control = np.load("/data/Control.npy")
    X = np.concatenate((PD, Control), axis=0)
    y = [1] * PD.shape[0]  + [0] * Control.shape[0]
    y = np.array(y)

    (x_train, y_train), (x_test, y_test) = ([], []), ([], [])

    rn = np.random.randint(100, size=614)
    for i in range(614):
        if(rn[i] < 70):
            x_train.append(X[i])
            y_train.append(y[i])
        else:
            x_test.append(X[i])
            y_test.append(y[i])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = (x_train.astype(np.float32) - 127.5)/127.5 #Scales from -1 to 1

    gen = generator(True, {'latent_dim': 88, 'strides':(2,2,2)})
    disc = discriminator(params={'shape':shap, 'strides':(2,2,2), 'kernel_size':(2,2,2), 'leak_value':0.25})

    model = Sequential()
    model.add(gen)
    disc.trainable = False
    model.add(disc)

    g_optimizer = Adam(lr = g_lr, beta_1 = beta)
    d_optimizer = Adam(lr = d_lr, beta_1 = beta)

    gen.compile(loss = 'binary_crossentropy', optimizer='SGD')
    model.compile(loss = 'binary_crossentropy', optimizer = g_optimizer)
    disc.trainable = True
    disc.compile(loss = 'binary_crossentropy', optimizer = d_optimizer)

    for epoch in range(epchs):
        num_steps = int(x_train.shape[0]/b_size)
        for index in range(num_steps):
            noise = np.random.uniform(-1, 1, size=(b_size, 88))
            image_batch = x_train[index*b_size:(index+1)*b_size]
            gen_img = gen.predict(noise, verbose = 0)
            X = np.concatenate((image_batch, gen_img))
            y = [1] * b_size + [0] * b_size
            d_l = disc.train_on_batch(X, y)
            noise = np.random.uniform(-1, 1, size=(b_size, 88))
            disc.trainable = False
            g_loss = model.train_on_batch(noise, [1] * b_size)
            disc.trainable = True
            if index % 10 == 9:
                gen.save_weights('generator', True)
                disc.save_weights('discriminator', True)

def retrain(b_size, epchs):
    PD = np.load("/data/PD.npy")
    Control = np.load("/data/Control.npy")
    X = np.concatenate((PD, Control), axis=0)
    y = [1] * PD.shape[0]  + [0] * Control.shape[0]
    y = np.array(y)

    (x_train, y_train), (x_test, y_test) = ([], []), ([], [])

    rn = np.random.randint(100, size=614)
    for i in range(614):
        if(rn[i] < 70):
            x_train.append(X[i])
            y_train.append(y[i])
        else:
            x_test.append(X[i])
            y_test.append(y[i])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = (x_train.astype(np.float32) - 127.5)/127.5 #Scales from -1 to 1

    gen = generator(True, {'latent_dim': 88, 'strides':(2,2,2)})
    disc = discriminator(params={'shape':shap, 'strides':(2,2,2), 'kernel_size':(2,2,2), 'leak_value':0.25})


    model = Sequential()
    model.add(gen)
    disc.trainable = False
    model.add(disc)

    g_optimizer = Adam(lr = g_lr, beta_1 = beta)
    d_optimizer = Adam(lr = d_lr, beta_1 = beta)

    gen.compile(loss = 'binary_crossentropy', optimizer='SGD')
    model.compile(loss = 'binary_crossentropy', optimizer = g_optimizer)
    disc.trainable = True
    disc.compile(loss = 'binary_crossentropy', optimizer = d_optimizer)
    gen.load_weights("generator")
    disc.load_weights("disciminator")

    for epoch in range(epchs):
        num_steps = int(x_train.shape[0]/b_size)
        for index in range(num_steps):
            noise = np.random.uniform(-1, 1, size=(b_size, 88))
            image_batch = x_train[index*b_size:(index+1)*b_size]
            gen_img = gen.predict(noise, verbose = 0)
            X = np.concatenate((image_batch, gen_img))
            y = [1] * b_size + [0] * b_size
            d_l = disc.train_on_batch(X, y)
            noise = np.random.uniform(-1, 1, size=(b_size, 88))
            disc.trainable = False
            g_loss = model.train_on_batch(noise, [1] * b_size)
            disc.trainable = True
            if index % 10 == 9:
                gen.save_weights('generator', True)
                disc.save_weights('discriminator', True)

def gen(num_images):
    gen = generator(True, {'latent_dim': 88, 'strides':(2,2,2), 'kernel_size':(5,5,5)})
    gen.compile(loss='binary_crossentropy', optimizer="SGD")
    gen.load_weights('generator')
    noise = np.random.uniform(-1, 1, (b_size, 88))
    generated_images = gen.predict(noise, verbose=1)
    image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(b_size = args.b_size, epchs = args.epochs)
    elif args.mode == "generate":
        gen( num_images = args.img)
    elif args.mode == "retrain":
        retrain(b_size = args.b_size, epchs = args.epochs)
