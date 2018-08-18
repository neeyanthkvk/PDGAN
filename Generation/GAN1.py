# From https://github.com/eriklindernoren/Keras-GAN

from __future__ import print_function, division

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

import numpy as np

def generator(phase_train=True, params={'latent_dim':200, 'strides':(2,2,2), 'kernel_size':(4,4,4)}):
    l_dim = params['latent_dim']
    strides = params['strides']
    kernel_size = params['kernel_size']

    inputs = Input(shape=(1,1,1,l_dim))

    


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
