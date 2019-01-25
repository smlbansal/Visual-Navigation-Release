import sys
import os
import pickle
import argparse


def convert_control_pipeline(control_pipeline_dir, py27_pipeline_dir):
    """
    Convert a control pipeline stored as a pickled object (via python3)
    into pickle files for use with python2.
    """
    filenames = os.listdir(control_pipeline_dir)
    filenames = list(filter(lambda x: 'pkl' in x, filenames))
    filenames.sort()
    for filename in filenames:
        print(filename)
        py3_filename = os.path.join(control_pipeline_dir, filename)
        py27_filename = os.path.join(py27_pipeline_dir, filename)
        with open(py3_filename, 'rb') as f:
            data_pickle = pickle.load(f)
        with open(py27_filename, 'wb') as f:
            pickle.dump(data_pickle, f, protocol=2)


def main(control_pipeline_dir):
    # Must be using python 3 to read and convert the pickle files
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    # Setup py3 and py27 control pipeline dirs
    assert(os.path.isdir(control_pipeline_dir))
    py27_pipeline_dir = os.path.join(control_pipeline_dir, 'py27')
    if not os.path.isdir(py27_pipeline_dir):
        os.mkdir(py27_pipeline_dir)

    # Convert the control pipeline
    convert_control_pipeline(control_pipeline_dir, py27_pipeline_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dir', type=str, required=True)
    args = parser.parse_args()
    main(control_pipeline_dir=args.dir)
