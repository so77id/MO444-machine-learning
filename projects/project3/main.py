from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

from utils.load_dataset import load_dataset,get_hash_ids

from methods.baseline import baseline
from methods.dbscan import dbscan
from methods.tsne import tsne




def main(argv):
     # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='input file', required=True)
    # parser.add_argument('-ds', '--descriptor_size', help='descriptor size', required=True)
    # parser.add_argument('-nd', '--n_descriptors', help='number of descriptors', required=True)
    parser.add_argument('-m', '--method', help='method', required=True)
    parser.add_argument('-a', '--ids', help='ids', required=True)
    ARGS = parser.parse_args()

    descriptors = load_dataset(ARGS.input_file)
    dict_ids = get_hash_ids(ARGS.ids)

    if ARGS.method == "baseline":
        baseline(descriptors,dict_ids)
    elif ARGS.method == "dbscan":
        dbscan(descriptors)
    elif ARGS.method == "tsne":
        tsne(descriptors)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))