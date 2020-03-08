from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil

import numpy as np
import tensorflow as tf
from absl import flags
from tqdm.autonotebook import tqdm

FLAGS = flags.FLAGS


def password_merge(passwords, n_rows=None, n_cols=None, padding=0, pad_value=0):
    passwords = (tf.cast(passwords, tf.float32) + 1.0) * 127.5
    passwords = np.array(passwords)
    n = passwords.shape[0]

    if n_rows:
        n_rows = max(min(n_rows, n), 1)
        n_cols = int(n - 0.5) // n_rows + 1
    elif n_cols:
        n_cols = max(min(n_cols, n), 1)
        n_rows = int(n - 0.5) // n_cols + 1
    else:
        n_rows = int(n ** 0.5)
        n_cols = int(n - 0.5) // n_rows + 1

    h, w = passwords.shape[1], passwords.shape[2]
    shape = (h * n_rows + padding * (n_rows - 1), w * n_cols + padding * (n_cols - 1))
    if passwords.ndim == 4:
        shape += (passwords.shape[3],)
    passw = np.full(shape, pad_value, dtype=passwords.dtype)

    for idx, password in enumerate(passwords):
        i = idx % n_cols
        j = idx // n_cols
        passw[j * (h + padding):j * (h + padding) + h, i * (w + padding):i *
                                                                       (w + padding) + w, ...] = password
    return passw


def save_password_grid(password_grid, epoch):
    file_name = FLAGS.dataset + f'_{epoch}.txt'
    output_dir = os.path.join(FLAGS.output_dir, file_name)
    tf.io.write_file(output_dir, tf.string.encode_utf8(tf.cast(password_grid, tf.uint8)))


def get_terminal_width():
    width = shutil.get_terminal_size(fallback=(200, 24))[0]
    if width == 0:
        width = 120
    return width


def pbar(total_passwords, batch_size, epoch, epochs):
    bar = tqdm(total=(total_passwords // batch_size) * batch_size,
               ncols=int(get_terminal_width() * .9),
               desc=tqdm.write(f'Epoch {epoch + 1}/{epochs}'),
               postfix={
                   'g_loss': f'{0:6.3f}',
                   'd_loss': f'{0:6.3f}',
                   1: 1
               },
               bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  '
                          'ETA: {remaining}  Elapsed Time: {elapsed}  '
                          'G Loss: {postfix[g_loss]}  D Loss: {postfix['
                          'd_loss]}',
               unit=' passwords',
               miniters=10)
    return bar