from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from absl import app
from absl import flags
from tensorflow import keras
from tensorflow.python.ops import control_flow_util

from models import WGANGP, DatasetPipeline

keras.backend.clear_session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 1, 'Epochs to train.')
flags.DEFINE_integer('batch_size', 64, 'Size of batch.')
flags.DEFINE_integer('layer_dim', 128, 'The hidden layer dimensionality for the generator.')
flags.DEFINE_integer('vocab_size', 257, 'dataset vocab size')
flags.DEFINE_integer('seq_len', 10, 'Sequence length for the passwords')
flags.DEFINE_integer('z_size', 128, 'Random vector noise size.')
flags.DEFINE_float('g_lr', .0001, 'Generator learning rate.')
flags.DEFINE_float('d_lr', .0001, 'Discriminator learning rate.')
flags.DEFINE_enum(
    'dataset', None,
    ['rock_you'],
    'Dataset to train.')
flags.DEFINE_boolean('preprocess', False, 'Pre-process the text data for normality.')
flags.DEFINE_string('output_dir', '.', 'Output directory.')
flags.DEFINE_float('g_penalty', 10.0, 'Gradient penalty weight.')
flags.DEFINE_integer('n_critic', 5, 'Critic updates per generator update.')
flags.DEFINE_integer('n_samples', 64, 'Number of samples to generate.')
flags.mark_flag_as_required('dataset')


def main(argv):
    del argv

    pipeline = DatasetPipeline()
    dataset = pipeline.load_dataset()

    wgangp = WGANGP(dataset_info=pipeline.dataset_info)
    wgangp.train(dataset=dataset)


if __name__ == '__main__':
    app.run(main)
