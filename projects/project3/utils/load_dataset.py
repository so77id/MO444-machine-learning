from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def load_dataset(file_name):
    with open(file_name, 'r') as file:
        descriptors = []

        for line in file:
            line = line.split(',')
            descriptors.append(np.array(line, dtype=np.float64))

        return np.array(descriptors, dtype=np.float64)