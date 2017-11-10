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
        return np.array(normalization(descriptors), dtype=np.float64)




# def parse_descriptors(argv):
# 	 # Parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input_file', help='input file', required=True)
#     parser.add_argument('-ds', '--descriptor_size', help='descriptor size', required=True)
#     parser.add_argument('-nd', '--n_descriptors', help='number of descriptors', required=True)
#     ARGS = parser.parse_args()


#     descriptors = load_dataset(ARGS.input_file)



def get_hash_ids(file_name, path='./dataset/docs'):

    with open(file_name, 'r') as file:
        ids = []
        news_groups = {}
        for idx, line in enumerate(file):
            id_ = line.split('\n')[0]
            ids.append(id_)
            # print("processing:", id_)
            with open("{}/{}".format(path,id_), 'r') as doc_in:

                for line in doc_in:
                    line = line.lower()
                    if "newsgroups" in line:
                        news_groups_line = line.split('\n')[0].split(' ')[-1].split(',')
                        news_groups[id_] = news_groups_line
                        break

    return np.array(ids), news_groups