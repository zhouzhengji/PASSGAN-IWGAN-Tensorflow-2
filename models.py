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
from tensorflow.python.keras import metrics

from ops import build_discriminator, build_generator, d_loss_fn, wasserstein_loss
from utils import pbar, save_password_grid, password_merge

tf.config.experimental_run_functions_eagerly(True)

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
        self.G = build_generator(layer_dim=self.layer_dim, seq_len=self.seq_len, vocab_size=self.vocab_size)
        self.D = build_discriminator(layer_dim=self.layer_dim, seq_len=self.seq_len, vocab_size=self.vocab_size)
        self.g_opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.g_lr, beta_1=0.5, beta_2=0.9)
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.d_lr, beta_1=0.5, beta_2=0.9)

    def train(self, dataset):
        z = tf.Variable(random.normal((FLAGS.n_samples, 2, self.z_dim)))
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()

        for epoch in tf.range(self.epochs):
            bar = pbar(self.total_passwords, self.batch_size, epoch, self.epochs)
            for batch in dataset:
                # tf.keras.backend.flatten(batch)
                for _ in tf.range(
                        self.n_critic):  # self.n_critic ##Tensor errors here, expects single tensor when is a list
                    text = batch['password']
                    text = tf.dtypes.cast(text, tf.float32)
                    real = tf.reshape(tf.Variable(text), [2, 2, 16])
                    self.train_d(real)
                    d_loss = self.train_d(real)
                    d_train_loss(d_loss)

                g_loss = self.train_g()
                g_train_loss(g_loss)
                self.train_g()

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.update(self.batch_size)

                if bar.n >= 1500:  # self.total_passwords * self.epochs:
                    break

            g_train_loss.reset_states()
            d_train_loss.reset_states()

            bar.close()
            del bar

            samples = self.generate_samples(z)
            pass_grid = password_merge(samples, n_rows=8).squeeze()
            # pass_grid = tfds.features.text.ByteTextEncoder.decode(self.vocab_size, ids=a.any(pass_grid))
            save_password_grid(pass_grid, epoch + 1)

    # def _tensorboard(self): TODO: Implement tensorboard
    #     tf.summary.scalar('loss/d_loss', self.d_loss)
    #     tf.summary.scalar('loss/g_loss', self.g_loss)
    #
    #     self.summary_op = tf.summary.merge_all()

    @tf.function
    def train_g(self):
        z = random.normal((self.batch_size, 1, self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            loss = wasserstein_loss(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    @tf.function  # This creates a variable on non-first, which tf.function refuses to do
    def train_d(self, real):
        with tf.GradientTape() as t:
            noise = tf.Variable(tf.random.normal((self.batch_size, 128)))
            x_fake = self.G(noise, training=True)
            fake_logits = self.D(x_fake, training=True)
            real_logits = self.D(real, reuse=True, training=True)
            cost = d_loss_fn(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), real, x_fake)
            cost += self.grad_penalty_weight * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost

    def gradient_penalty(self, f, real, fake):
        real = tf.reshape(real, [2, 1, 32])
        alpha = random.uniform([2, 1], 0., 1.)
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
                                          split=tfds.Split.TRAIN,  # Variations of split are test, train or validation
                                          with_info=True)
        # ds = ds.map(lambda x: self.preprocess_password(x['password']), AUTOTUNE)
        ds = self.dataset_cache(ds)
        ds = ds.shuffle(50000, reshuffle_each_iteration=True)
        ds = ds.apply(tf.data.experimental.unbatch())
        ds = ds.batch(self.batch_size, drop_remainder=True)  # .prefetch(AUTOTUNE)
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
