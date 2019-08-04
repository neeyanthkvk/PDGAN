import numpy as np
from keras import backend as K
from keras import layers
from keras import models
from keras.engine import InputSpec
from keras.initializers import Constant


class Sparse(layers.Dense):
    def __init__(self,
                 adjacency_mat=None,
                 # Specifies which inputs (rows) are connected to which outputs
                 # (columns)
                 *args,
                 **kwargs):
        self.adjacency_mat = adjacency_mat
        if adjacency_mat is not None:
            units = adjacency_mat.shape[1]
            super().__init__(units=units, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        connection_vector = (
            np.sum(self.adjacency_mat, axis=0) > 0
        ).astype(int)
        if np.sum(connection_vector) < self.adjacency_mat.shape[1]:
            print('Warning: not all nodes in the Sparse layer are ' +
                  'connected to inputs! These nodes will always have zero ' +
                  'output.')

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=lambda shape: K.constant(
                self.adjacency_mat) *
            self.kernel_initializer(shape),

            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=lambda shape: K.constant(connection_vector) *
                self.bias_initializer(shape),

                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

            self.bias_adjacency_tensor = self.add_weight(
                shape=(self.units,),
                initializer=Constant(connection_vector),
                name='bias_adjacency_matrix',
                trainable=False)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        # Ensure we set weights to zero according to adjancency matrix
        self.adjacency_tensor = self.add_weight(
            shape=(input_dim, self.units),
            initializer=Constant(self.adjacency_mat),
            name='adjacency_matrix',
            trainable=False)
        self.built = True

    def call(self, inputs):
        output = self.kernel * self.adjacency_tensor
        output = K.dot(inputs, output)
        if self.use_bias:
            output = K.bias_add(output, self.bias_adjacency_tensor * self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def count_params(self):
        num_weights = 0
        if self.use_bias:
            bias_weights = np.sum(
                (np.sum(self.adjacency_mat, axis=0) > 0).astype(int))
            num_weights += bias_weights
        num_weights += np.sum(self.adjacency_mat)
        return num_weights

    def get_config(self):
        config = {
            'adjacency_mat': self.adjacency_mat.tolist()
        }
        base_config = super().get_config()
        base_config.pop('units', None)
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        adjacency_mat_as_list = config['adjacency_mat']
        config['adjacency_mat'] = np.array(adjacency_mat_as_list)
        return cls(**config)


def add_common_layers(y):
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    return y


def grouped_convolution(y, nb_channels, strides):
    return layers.Conv3D(nb_channels, kernel_size=(3, 3), strides=strides,
                         padding='same')(y)


def residual_block(y, nb_channels_in, nb_channels_out, strides=(1, 1),
                   project_shortcut=False):
    shortcut = y
    y = layers.Conv3D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1),
                      padding='same')(y)
    y = add_common_layers(y)
    y = grouped_convolution(y, nb_channels_in, strides=strides)
    y = add_common_layers(y)
    y = layers.Conv3D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1),
                      padding='same')(y)
    y = layers.BatchNormalization()(y)
    if project_shortcut or strides != (1, 1):
        shortcut = layers.Conv3D(nb_channels_out, kernel_size=(1, 1),
                                 strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)
    return y


def create_model(inp_shape):
    inp = layers.Input(shape=inp_shape)
    x = layers.Conv3D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inp)
    x = add_common_layers(x)

    x = layers.MaxPool3D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, project_shortcut=project_shortcut)

    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, strides=strides)

    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, strides=strides)

    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 1024, 2048, strides=strides)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(16)(x)
    x = Sparse(np.random.randint(16, size=10).reshape((16, 2)))(x)
    x = layers.Dense(1)(x)

    model = models.Model(inputs=[inp], outputs=[x])
    print(model.summary())
    return model
