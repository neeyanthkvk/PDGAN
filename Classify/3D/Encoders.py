import keras
import numpy as np
import keras.backend as K
from sklearn.model_selection import train_test_split

# Model Creation
from keras.models import Sequential, Model
from keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D, GlobalAveragePooling3D
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dropout, Dense, Reshape, BatchNormalization, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import plot_model


input_shape = (176, 256, 240, 1)

input_img = Input(shape=input_shape)

encoder = (Conv3D(32, kernel_size=(7, 7, 7), input_shape=(176, 256, 240, 1), padding="same"))(input_img)
encoder = (LeakyReLU())(encoder)
encoder = (Conv3D(32, padding="same", kernel_size=(7, 7, 7)))(encoder)
encoder = (LeakyReLU())(encoder)
encoder = (MaxPooling3D(pool_size=(2, 2, 2), padding="same"))(encoder)
encoder = (Dropout(0.25))(encoder)

encoder = (Conv3D(128, padding="same", kernel_size=(5, 5, 5)))(encoder)
encoder = (LeakyReLU())(encoder)
encoder = (Conv3D(128, padding="same", kernel_size=(5, 5, 5)))(encoder)
encoder = (LeakyReLU())(encoder)
encoder = (MaxPooling3D(pool_size=(2, 2, 2), padding="same"))(encoder)
encoder = (Dropout(0.25))(encoder)

encoder = (Conv3D(64, padding="same", kernel_size=(5, 5, 5)))(encoder)
encoder = (LeakyReLU())(encoder)
encoder = (Conv3D(64, padding="same", kernel_size=(5, 5, 5)))(encoder)
encoder = (LeakyReLU())(encoder)
encoder = (MaxPooling3D(pool_size=(2, 2, 2), padding="same"))(encoder)
encoder = (Dropout(0.25))(encoder)

encoder = (Conv3D(32, padding="same", kernel_size=(3, 3, 3)))(encoder)
encoder = (LeakyReLU())(encoder)
encoder = (Conv3D(32, padding="same", kernel_size=(3, 3, 3)))(encoder)
encoder = (LeakyReLU())(encoder)
encoder = (MaxPooling3D(pool_size=(2, 2, 2), padding="same"))(encoder)
encoder = (Dropout(0.25))(encoder)

encoder = (Conv3D(8, padding="same", kernel_size=(3, 3, 3)))(encoder)
encoder = (LeakyReLU())(encoder)
encoder = (Conv3D(8, padding="same", kernel_size=(3, 3, 3)))(encoder)
encoder = (LeakyReLU())(encoder)
encoder = (MaxPooling3D(pool_size=(2, 2, 2), padding="same"))(encoder)
encoder = (Dropout(0.25))(encoder)

decoder = (Conv3D(8, padding="same", kernel_size=(3, 3, 3)))(encoder)
decoder = (LeakyReLU())(decoder)
decoder = (UpSampling2D(pool_size=(2, 2, 2), padding="same"))(decoder)
decoder = (Conv3D(8, padding="same", kernel_size=(3, 3, 3)))(decoder)
decoder = (LeakyReLU())(decoder)
decoder = (Dropout(0.25))(decoder)

decoder = (Conv3D(32, padding="same", kernel_size=(3, 3, 3)))(decoder)
decoder = (LeakyReLU())(decoder)
decoder = (UpSampling2D(pool_size=(2, 2, 2), padding="same"))(decoder)
decoder = (Conv3D(32, padding="same", kernel_size=(3, 3, 3)))(decoder)
decoder = (LeakyReLU())(decoder)
decoder = (Dropout(0.25))(decoder)

decoder = (Conv3D(64, padding="same", kernel_size=(5, 5, 5)))(decoder)
decoder = (LeakyReLU())(decoder)
decoder = (UpSampling2D(pool_size=(2, 2, 2), padding="same"))(decoder)
decoder = (Conv3D(64, padding="same", kernel_size=(5, 5, 5)))(decoder)
decoder = (LeakyReLU())(decoder)
decoder = (Dropout(0.25))(decoder)

decoder = (Conv3D(128, padding="same", kernel_size=(5, 5, 5)))(decoder)
decoder = (LeakyReLU())(decoder)
decoder = (UpSampling2D(pool_size=(2, 2, 2), padding="same"))(decoder)
decoder = (Conv3D(128, padding="same", kernel_size=(5, 5, 5)))(decoder)
decoder = (LeakyReLU())(decoder)
decoder = (Dropout(0.25))(decoder)

decoder = (Conv3D(32, padding="same", kernel_size=(7, 7, 7)))(decoder)
decoder = (LeakyReLU())(decoder)
decoder = (UpSampling2D(pool_size=(2, 2, 2), padding="same"))(decoder)
decoder = (Conv3D(32, padding="same", kernel_size=(7, 7, 7)))(decoder)
decoder = (LeakyReLU())(decoder)
decoder = (Dropout(0.25))(decoder)

decoder = (Conv3D(1, padding="same", kernel_size=(7, 7, 7)))(decoder)

autoencoder = Model(input_img, decoder)
