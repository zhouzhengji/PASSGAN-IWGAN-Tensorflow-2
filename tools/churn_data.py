import random
import csv
import numpy as np

random.seed(5421)

with open('../data/rockyou1.tsv', 'r', newline='') as f:
    print('imported file')

    lines = csv.reader(f, delimiter=',')

    print('removing excess characters')
    lines = list(filter(lambda x: len(x) <= 10, lines))

    print('randomising')
    random.shuffle(lines)

    train, validate, test = np.split(lines, [int(len(lines) * 0.8), int(len(lines) * 0.9)])

    with open('../data/train.tsv', 'w') as f:
        for item in train:
            f.writelines('\n'.join(item))
            f.write("\n")
            print('training data written')

    with open('../data/test.tsv', 'w') as f:
        for item in test:
            f.writelines('\n'.join(item))
            f.write("\n")
            print('test data written')

    with open('../data/validate.tsv', 'w') as f:
        for item in validate:
            f.writelines('\n'.join(item))
            f.write("\n")
            print('validation data written')

