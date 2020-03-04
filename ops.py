from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import self as self
from tensorflow import optimizers
from tensorflow import reduce_mean
from tensorflow.python.keras import layers
import tensorflow as tf
from tensorflow_core.python.keras.layers import noise

import utils


class Conv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same'):
        super(Conv2D, self).__init__()
        self.conv_op = layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     use_bias=False,
                                     kernel_initializer='he_normal')

    def call(self, inputs, **kwargs):
        return self.conv_op(inputs)


class UpConv2D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same'):
        super(UpConv2D, self).__init__()
        self.up_conv_op = layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 use_bias=False,
                                                 kernel_initializer='he_normal')

    def call(self, inputs, **kwargs):
        return self.up_conv_op(inputs)


class Conv1D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same'):
        super(Conv1D, self).__init__()
        self.conv_op = layers.Conv1D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     use_bias=False,
                                     kernel_initializer='he_normal')

    def call(self, inputs, **kwargs):
        return self.conv_op(inputs)


class UpConv1D(layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same'):
        super(UpConv1D, self).__init__()
        self.up_conv_op = utils.Conv1DTranspose(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding=padding)

    # def call(self, inputs, **kwargs):
    #     return self.up_conv_op(inputs)


class BatchNorm(layers.Layer):
    def __init__(self, epsilon=1e-4, axis=-1, momentum=0.99):
        super(BatchNorm, self).__init__()
        self.batch_norm = layers.BatchNormalization(epsilon=epsilon,
                                                    axis=axis,
                                                    momentum=momentum)

    def call(self, inputs, **kwargs):
        return self.batch_norm(inputs)


class LayerNorm(layers.Layer):
    def __init__(self, epsilon=1e-4, axis=-1):
        super(LayerNorm, self).__init__()
        self.layer_norm = layers.LayerNormalization(epsilon=epsilon, axis=axis)

    def call(self, inputs, **kwargs):
        return self.layer_norm(inputs)


class LeakyRelu(layers.Layer):
    def __init__(self, alpha=0.2):
        super(LeakyRelu, self).__init__()
        self.leaky_relu = layers.LeakyReLU(alpha=alpha)

    def call(self, inputs, **kwargs):
        return self.leaky_relu(inputs)


class AdamOptWrapper(optimizers.Adam):
    def __init__(self,
                 learning_rate=1e-4,
                 beta_1=0.,
                 beta_2=0.9,
                 epsilon=1e-4,
                 amsgrad=False,
                 **kwargs):
        super(AdamOptWrapper, self).__init__(learning_rate, beta_1, beta_2, epsilon,
                                             amsgrad, **kwargs)


def d_loss_fn(f_logit, r_logit):
    f_loss = reduce_mean(f_logit)
    r_loss = reduce_mean(r_logit)
    return f_loss - r_loss


def g_loss_fn(f_logit):
    f_loss = -reduce_mean(f_logit)
    return f_loss


class ResBlock():
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res_block = tf.keras.Sequential(
            tf.keras.activations.relu(True),
            tf.keras.layers.Conv1D(dim, dim, 5, padding=2),  # nn.Linear(DIM, DIM),
            tf.keras.activations.relu(True),
            tf.keras.layers.Conv1D(dim, dim, 5, padding=2),  # nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3 * output)


class BuildGenerator(self):
    def __init__(self):
        super(BuildGenerator, self).__init__()
        dim = self.layer_dim
        seq_len = self.seq_len
        vocab_size = self.vocab_size

        self.fc1 = tf.keras.activations.linear(128, dim * seq_len)
        self.block = tf.keras.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.conv1d = tf.keras.layers.Conv1D(self.vocab_size, dim, 1)
        self.softmax = tf.keras.activations.softmax()

    def forward(self, noise):
        batch_size = noise.size(0)
        output = self.fc1(noise)
        # (BATCH_SIZE, DIM, SEQ_LEN)
        output = output.view(-1, self.dim, self.seq_len)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(batch_size * self.seq_len, -1)
        output = self.softmax(output)
        # (BATCH_SIZE, SEQ_LEN, len(charmap))
        return output.view(shape)


class BuildDiscriminator(self):
    def __init__(self):
        super(BuildDiscriminator, self).__init__()
        dim = self.layer_dim
        seq_len = self.seq_len
        vocab_size = self.vocab_size

        self.fc1 = tf.keras.activations.linear(128, dim * seq_len)
        self.block = tf.keras.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.conv1d = tf.keras.layers.Conv1D(self.vocab_size, dim, 1)
        self.linear = tf.keras.activations.linear(seq_len * dim, 1)

    def forward(self, input):
        output = input.transpose(1, 2)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len * self.layer_dim)
        output - self.linear(output)
        return output
