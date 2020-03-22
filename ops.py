from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import reduce_mean


def d_loss_fn(f_logit, r_logit):
    f_loss = tf.math.reduce_mean(f_logit)
    r_loss = tf.math.reduce_mean(r_logit)
    return f_loss - r_loss


def wasserstein_loss(f_logit):
    f_loss = -reduce_mean(f_logit)
    return f_loss


class ResBlock(tf.keras.Model):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res_block = tf.keras.Sequential([
            # tf.keras.layers.Dense(128, activation='linear', input_shape=(784,1)),
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 6, padding='same'),  # (dim, dim, 5, padding='same')
            tf.keras.layers.ReLU(True),
            tf.keras.layers.Conv1D(dim, dim, 6, padding='same'),
        ])

    def call(self, input, **kwargs):
        output = self.res_block(input)
        return input + (0.3 * output)


class build_generator(tf.keras.Model):
    def __init__(self, layer_dim, seq_len, vocab_size):
        super(build_generator, self).__init__()
        dim = layer_dim
        self.dim = layer_dim
        self.seq_len = seq_len

        self.fc1 = tf.keras.layers.Dense(128, activation='linear', input_shape=(dim * seq_len,))
        self.block = tf.keras.Sequential([
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        ])
        self.conv1 = tf.keras.layers.Conv1D(1, 1, 1)  # (dim, vocab_size, 1)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, noise, **kwargs):
        batch_size = tf.size(noise)  # tf.keras.layers.GaussianNoise(0, input_shape=(self.dim, self.seq_len))
        output = self.fc1(noise)
        output = tf.reshape(output, (-1, 32, 128))  # (-1, 32, 128))   (-1, self.dim, self.seq_len)
        output = self.block(output)
        output = self.conv1(output)
        output = tf.transpose(output, [0, 2, 1])
        shape = tf.size(output)
        # output = tf.contiguous()
        # output = tf.reshape(output, (batch_size, 2, -1))
        output = self.softmax(output)
        return output  # tf.reshape(output, shape)


class build_discriminator(tf.keras.Model):
    def __init__(self, layer_dim, seq_len, vocab_size):
        super(build_discriminator, self).__init__()
        dim = layer_dim
        self.dim = layer_dim
        self.seq_len = seq_len
        vocab_size = vocab_size

        self.block = tf.keras.Sequential([
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        ])
        self.conv1d = tf.keras.layers.Conv1D(dim, 1, 1)
        self.linear = tf.keras.layers.Dense(seq_len * dim, activation='linear')

    def call(self, input, **kwargs):
        output = tf.transpose(input)
        output = self.conv1d(output)
        output = self.block(output)
        output = tf.reshape(output, (-1, 32, 128))
        output = self.linear(output)
        return output
#
