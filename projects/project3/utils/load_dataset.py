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




# def parse_descriptors(argv):
# 	 # Parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input_file', help='input file', required=True)
#     parser.add_argument('-ds', '--descriptor_size', help='descriptor size', required=True)
#     parser.add_argument('-nd', '--n_descriptors', help='number of descriptors', required=True)
#     ARGS = parser.parse_args()


#     descriptors = load_dataset(ARGS.input_file)



def get_hash_ids(file_name):
    with open(file_name, 'r') as file:
        dict_hash_ids={}
        for idx, line in enumerate(file):
            dict_hash_ids[idx] = line


    return dict_hash_ids