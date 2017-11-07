from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from utils.normalization import normalization

def load_dataset(file_name):
    print("Loading dataset")
    with open(file_name, 'r') as file:
        descriptors = []

        for line in file:
            line = line.split(',')
            descriptors.append(np.array(line, dtype=np.float64))

        descriptors = np.matrix(descriptors, dtype=np.float64)

        return normalization(descriptors)
