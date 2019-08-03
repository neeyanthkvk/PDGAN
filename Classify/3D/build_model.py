from keras import layers
from keras import models

img_shape = (128, 128, 128)


def batch_relu(inp):
    inp = layers.BatchNormalization()(inp)
    inp = layers.LeakyReLU()(inp)
    return inp


def conv(inp, num_channels, strides):
    return layers.Conv3D(num_channels, kernel_size=(3, 3, 3),
                         strides=strides, padding='same')(inp)


def residual_block(inp, ):
    # TODO: Implement Method
    cop = inp
    inp = layers.Conv2D()(inp)
    inp = layers.Conv2D()(inp)

    pass


def create_model():
    # TODO: Implement Method
    pass
