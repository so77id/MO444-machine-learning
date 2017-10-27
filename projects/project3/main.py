import sys
import argparse
import numpy as np

from utils.load_dataset import load_dataset

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='input file', required=True)
    parser.add_argument('-ds', '--descriptor_size', help='descriptor size', required=True)
    parser.add_argument('-nd', '--n_descriptors', help='number of descriptors', required=True)
    ARGS = parser.parse_args()


    descriptors = load_dataset(ARGS.input_file)





if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))