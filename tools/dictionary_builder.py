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
import csv
import sys
import pathlib

import tensorflow as tf
import tensorflow_datasets as tfds

tf.keras.backend.clear_session()
csv.field_size_limit(sys.maxsize)


class DatasetPipeline:
    def __init__(self):
        self.dataset_name = 'rock_you'
        self.batch_size = 64
        self.dataset_info = []
        with open('../data/test.tsv', encoding="utf8") as f:
            self.csvreader = csv.reader(f, delimiter=",")
            self.lines = list(self.csvreader)

    def load_dataset(self):
        encoded = []
        original = []
        ds, self.dataset_info = tfds.load(name=self.dataset_name, split=None, with_info=True)
        ds = tfds.as_numpy(ds)

        self.encoder = self.dataset_info.features['password'].encoder
        print('Vocabulary size: {}'.format(self.encoder.vocab_size))

        for elem in str(self.lines):
            encoded_string = self.encoder.encode(elem)
            print('Encoded string is {}'.format(encoded_string))
            encoded.append(encoded_string)

            original_string = [self.encoder.decode(encoded_string)]
            print('The original string: "{}"'.format(original_string))
            original.append(original_string)

        with open('../data/dictionary.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(encoded, original))

        return ds


pipeline = DatasetPipeline()
dataset = pipeline.load_dataset()
samples = pathlib.Path('../data/samples.txt')

# if samples.exists():
#     print("Samples Exist, Beginning decoding")
#     with open('../data/samples.txt', encoding="utf8") as f:
#         csvreader = csv.reader(f, delimiter="/n")
#         sample_int = int(csvreader)
#         print(chr(sample_int))

