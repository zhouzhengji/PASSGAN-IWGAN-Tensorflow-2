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

import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.summary import summary
from tensorflow.python.tools import saved_model_utils

keras.backend.clear_session()
output = []
model_dir = 'models/generator/rock_you20200428-205637'
log_dir = 'models/generator/'
tag_set = 'serve'

available_model = tf.saved_model.contains_saved_model('models/generator/rock_you20200428-205637')
print(available_model)

model = tf.saved_model.load(
    'models/generator/rock_you20200428-205637')
for i in range(200):
    z = np.random.randint(258, size=(128), dtype=np.int64)
    z = tf.dtypes.cast(z, tf.float32)
    z = tf.reshape(tf.Variable(z), [2, 1, 64])
    samples = model(z, training=False)
    raw = tf.nest.flatten(z.numpy().tolist())
    probability = tf.nest.flatten(samples.numpy().tolist())

    intersection = [i for i, j in zip(raw, probability) if 1.0 == j]
    output.append(intersection)

output = list(filter(None, output))
np.savetxt('data/samples.txt', output, fmt="%10s")

# Decodes samples
output = [[int(float(j)) for j in i] for i in output]
output = str(output).replace('[', '').replace(']', '').replace(' ', '')
output = [int(x) for x in output.split(',') if x.strip().isdigit()]
charList = [chr(output[i]) for i in range(0, len(output))]
np.savetxt('data/decoded_samples.txt', charList, fmt="%10s")
print(charList)


# Displays interactive model flow in tensorboard
def import_to_tensorboard(model_dir, log_dir, tag_set):
    with session.Session(graph=ops.Graph()) as sess:
        input_graph_def = saved_model_utils.get_meta_graph_def(model_dir,
                                                               tag_set).graph_def
        importer.import_graph_def(input_graph_def)

        pb_visual_writer = summary.FileWriter(log_dir)
        pb_visual_writer.add_graph(sess.graph)
        print("Model Imported. Visualize by running: "
              "tensorboard --logdir={}".format(log_dir))


import_to_tensorboard(model_dir, log_dir, tag_set)
