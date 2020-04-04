from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags
from tensorflow import random
from tensorflow.python.keras import metrics
import numpy as np

from ops import build_discriminator, build_generator, d_loss_fn, wasserstein_loss
from utils import pbar

tf.config.experimental_run_functions_eagerly(True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS


class WGANGP:
    def __init__(self, dataset_info):
        self.dataset_name = FLAGS.dataset
        self.z_dim = FLAGS.z_size
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.seq_len = FLAGS.seq_len
        self.layer_dim = FLAGS.layer_dim
        self.n_critic = FLAGS.n_critic
        self.vocab_size = FLAGS.vocab_size
        self.grad_penalty_weight = FLAGS.g_penalty
        self.total_passwords = dataset_info.splits.total_num_examples
        self.encoder = dataset_info.features['password'].encoder
        self.G = build_generator(layer_dim=self.layer_dim, seq_len=self.seq_len, vocab_size=self.vocab_size)
        self.D = build_discriminator(layer_dim=self.layer_dim, seq_len=self.seq_len, vocab_size=self.vocab_size)
        self.g_opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.g_lr, beta_1=0.5, beta_2=0.9)
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.d_lr, beta_1=0.5, beta_2=0.9)

    def train(self, dataset):
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for epoch in tf.range(self.epochs):
            epoch = tf.cast(epoch, dtype=tf.int64, name=epoch)
            bar = pbar(self.total_passwords, self.batch_size, epoch, self.epochs)
            for batch in dataset:
                for _ in tf.range(
                        self.n_critic):
                    self.text = batch['password']
                    self.text = tf.dtypes.cast(self.text, tf.float32)
                    real = tf.reshape(tf.Variable(self.text), [2, 1, 32])
                    # samples = real
                    # samples = tf.cast(samples, dtype=tf.int32)
                    # samples = np.array([samples]).flatten().tolist()
                    # decoded_passwords = self.encoder.decode(samples)
                    # print(decoded_passwords)
                    self.train_d(real)
                    d_loss = self.train_d(real)
                    d_train_loss(d_loss)

                g_loss = self.train_g()
                g_train_loss(g_loss)
                self.train_g()
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', g_train_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', d_train_loss.result(), step=epoch)

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.update(self.batch_size)

                if bar.n >= 20000:  # self.total_passwords * self.epochs:
                    break

            self.G.summary()
            self.D.summary()
            tf.saved_model.save(self.G, './models/generator/' + self.dataset_name + current_time)
            tf.saved_model.save(self.D, './models/discriminator/' + self.dataset_name + current_time)

            g_train_loss.reset_states()
            d_train_loss.reset_states()

            bar.close()
            del bar

            z = np.random.randint(258, size=(128, 128), dtype=np.int64)
            z = tf.dtypes.cast(z, tf.float32)
            z = tf.reshape(tf.Variable(z), [128, 128])
            samples = (self.generate_samples(z))
            with open("samples.txt" + current_time, "w") as output:
                output.write(str(samples))



    @tf.function
    def train_g(self):

        z = np.random.randint(258, size=(128, 128), dtype=np.int64)
        z = tf.dtypes.cast(z, tf.float32)
        z = tf.reshape(tf.Variable(z), [128, 128])
        with tf.GradientTape() as t:
            t.watch(z)
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            loss = wasserstein_loss(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    @tf.function
    def train_d(self, real):
        z = np.random.randint(258, size=(128, 128), dtype=np.int64)
        z = tf.dtypes.cast(z, tf.float32)
        z = tf.reshape(tf.Variable(z), [128, 128])
        with tf.GradientTape() as t:
            t.watch(z)
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            real_logits = self.D(real, reuse=True, training=True)
            cost = d_loss_fn(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), real, x_fake)
            cost += self.grad_penalty_weight * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost

    def gradient_penalty(self, f, real, fake):
        # fake = tf.reshape(fake, [2, 1, 64])
        real = tf.tile(real, multiples=[2, 1, 1])
        alpha = random.uniform([2, 1], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        inter = tf.reshape(inter, [8, 1, 32])
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    @tf.function
    def generate_samples(self, z):
        # Generates samples using random values from a Gaussian distribution.
        return self.G(z, training=False)


class DatasetPipeline:
    def __init__(self):
        self.dataset_name = FLAGS.dataset
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.layer_dim = FLAGS.layer_dim
        self.preprocess = FLAGS.preprocess
        self.dataset_info = []

    def preprocess_label(self, passwords):  # TODO: tie in preprocess tools for dataset
        return tf.cast(passwords, tf.int64)

    def dataset_cache(self, dataset):
        tmp_dir = Path(tempfile.gettempdir())
        cache_dir = tmp_dir.joinpath('cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        for p in cache_dir.glob(self.dataset_name + '*'):
            p.unlink()
        return dataset.cache(str(cache_dir / self.dataset_name))

    def load_dataset(self):
        ds, self.dataset_info = tfds.load(name=self.dataset_name,
                                          split=tfds.Split.TRAIN,
                                          with_info=True)

        ds = self.dataset_cache(ds)
        ds = ds.shuffle(50000, reshuffle_each_iteration=True)
        ds = ds.apply(tf.data.experimental.unbatch())
        ds = ds.batch(self.batch_size, drop_remainder=True)
        return ds
