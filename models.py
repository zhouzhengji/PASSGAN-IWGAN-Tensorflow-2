from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
from functools import partial
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags
from tensorflow import random
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models

import ops
from utils import img_merge
from utils import pbar
from utils import save_image_grid

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS


class WGANGP:
    def __init__(self, dataset_info):
        self.z_dim = FLAGS.z_size
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.seq_len = FLAGS.seq_len
        self.layer_dim = FLAGS.layer_dim
        self.n_critic = FLAGS.n_critic
        self.vocab_size = FLAGS.vocab_size
        self.grad_penalty_weight = FLAGS.g_penalty
        self.total_passwords = dataset_info.splits.total_num_examples
        self.g_opt = ops.AdamOptWrapper(learning_rate=FLAGS.g_lr)
        self.d_opt = ops.AdamOptWrapper(learning_rate=FLAGS.d_lr)
        self.G = ops.BuildGenerator()
        self.D = ops.BuildDiscriminator()

        self.G.summary()
        self.D.summary()

    def train(self, dataset):
        z = tf.constant(random.normal((FLAGS.n_samples, 2, self.z_dim)))
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()

        for epoch in range(self.epochs):
            bar = pbar(self.total_passwords, self.batch_size, epoch, self.epochs)
            for batch in dataset:
                for _ in range(self.n_critic):
                    self.train_d(batch)
                    d_loss = self.train_d(batch)
                    d_train_loss(d_loss)

                g_loss = self.train_g()
                g_train_loss(g_loss)
                self.train_g()

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.update(self.batch_size)

            g_train_loss.reset_states()
            d_train_loss.reset_states()

            bar.close()
            del bar

            samples = self.generate_samples(z)
            image_grid = img_merge(samples, n_rows=8).squeeze()
            save_image_grid(image_grid, epoch + 1)

    @tf.function
    def train_g(self):
        z = random.normal((self.batch_size, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            loss = ops.g_loss_fn(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    @tf.function
    def train_d(self, x_real):
        with tf.GradientTape() as t:
            z = random.normal((self.batch_size, self.z_dim, 128))  # HERE
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            real_logits = self.D(x_real, training=True)
            cost = ops.d_loss_fn(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), x_real, x_fake)
            cost += self.grad_penalty_weight * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost

    def gradient_penalty(self, f, real, fake):
        alpha = random.uniform([self.batch_size, 1], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
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

        # while mult > 1:
        #     x = ops.UpConv1D(dim * (mult // 2))(x)
        #     x = ops.BatchNorm()(x)
        #     x = layers.ReLU()(x)
        #
        #     mult //= 2
        #
        # x = ops.UpConv1D(3)(x)
        # x = layers.Activation('tanh')(x)
        # return models.Model(inputs, x, name='Generator')

        # x = inputs = layers.Input((dim, dim))
        # x = ops.Conv1D(dim)(x)
        # x = ops.LeakyRelu()(x)
        #
        # while i > 4:
        #     x = ops.Conv1D(dim * (2 * mult))(x)
        #     x = ops.LayerNorm(axis=[1, 2])(x)
        #     x = ops.LeakyRelu()(x)
        #
        #     i //= 2
        #     mult *= 2
        #
        # x = ops.Conv1D(1, 4, 1, 'valid')(x)
        # return models.Model(inputs, x, name='Discriminator')


class DatasetPipeline:
    def __init__(self):
        self.dataset_name = FLAGS.dataset
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.layer_dim = FLAGS.layer_dim
        self.preprocess = FLAGS.preprocess
        self.dataset_info = {}

    # def preprocess_password(self, password):  #TODO: tie in preprocess tools for dataset
    #     if self.preprocess:
    #         password = do_tools_process
    #     return password

    def dataset_cache(self, dataset):
        tmp_dir = Path(tempfile.gettempdir())
        cache_dir = tmp_dir.joinpath('cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        for p in cache_dir.glob(self.dataset_name + '*'):
            p.unlink()
        return dataset.cache(str(cache_dir / self.dataset_name))

    def load_dataset(self):
        ds, self.dataset_info = tfds.load(name=self.dataset_name,
                                          split=tfds.Split.TRAIN,  # Variations of split are test, train or validation
                                          with_info=True)
        # ds = ds.map(lambda x: self.preprocess_password(x['password']), AUTOTUNE)
        ds = self.dataset_cache(ds)
        ds = ds.shuffle(50000, reshuffle_each_iteration=True)
        ds = ds.apply(tf.data.experimental.unbatch())
        ds = ds.batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        return ds

    # def load_dataset(self, dl_manager):
    #     # Download source data
    #     extracted_path = dl_manager.download_and_extract(dl_manager.download('http://www.digitalviolence.co.uk/rockyou1.tsv'))
    #
    #     # Specify the splits
    #     return [
    #         tfds.core.SplitGenerator(
    #             name=tfds.Split.TRAIN,
    #             gen_kwargs={
    #                 "text_dir_path": os.path.join(extracted_path, "train"),
    #                 "passwords": os.path.join(extracted_path, "train_passwords.tsv"),
    #             },
    #         ),
    #         tfds.core.SplitGenerator(
    #             name=tfds.Split.TEST,
    #             gen_kwargs={
    #                 "text_dir_path": os.path.join(extracted_path, "test"),
    #                 "passwords": os.path.join(extracted_path, "test_passwords.tsv"),
    #             },
    #         ),
    #     ]
