import argparse
from glob import glob
from os import environ
from os.path import join as pjoin
from shutil import copy
from sys import platform

def main():
    """
    Fix paths to dlls
    """
    p = argparse.ArgumentParser()
    p.add_argument("sitepackagesdir")
    args = p.parse_args()
    hdf5_path = environ.get("HDF5_DIR")
    if platform.startswith('win'):
        for f in glob(pjoin(hdf5_path, 'lib/*.dll')):
            copy(f, pjoin(args.sitepackagesdir, 'h5py'))


if __name__ == '__main__':
    main()
