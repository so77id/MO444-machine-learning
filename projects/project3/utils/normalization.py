from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def normalization(descriptors):
    descriptors = np.matrix(descriptors)
    descriptors = descriptors - descriptors.mean(axis=1)
    return descriptors/descriptors.std(axis=1)
