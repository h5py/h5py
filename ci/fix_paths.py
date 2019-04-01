import distutils.sysconfig
from glob import glob
from os import environ
from os.path import join as pjoin
from shutil import copy
from sys import platform

def main():
    """
    Copy HDF5 DLLs into installed h5py package
    """
    # This is the function Tox also uses to locate site-packages (Apr 2019)
    sitepackagesdir = distutils.sysconfig.get_python_lib()

    hdf5_path = environ.get("HDF5_DIR")
    if platform.startswith('win'):
        for f in glob(pjoin(hdf5_path, 'lib/*.dll')):
            copy(f, pjoin(sitepackagesdir, 'h5py'))


if __name__ == '__main__':
    main()
