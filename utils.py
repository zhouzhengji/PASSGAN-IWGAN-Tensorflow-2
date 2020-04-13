#  PassGAN_Final_Year_Project - Replication of PassGAN paper using Tensorflow 2 & Keras
#  Copyright (C) 2020 RachelaHorner
#
#  This file is part of PassGAN_Final_Year_Project (PFYP).
#
#  PFYP is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  PFYP is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with PFYP.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import shutil

from absl import flags
from tqdm.autonotebook import tqdm

FLAGS = flags.FLAGS


def get_terminal_width():
    width = shutil.get_terminal_size(fallback=(200, 24))[0]
    if width == 0:
        width = 120
    return width


def pbar(total_passwords, batch_size, epoch, epochs):
    bar = tqdm(total=total_passwords * epochs,
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
