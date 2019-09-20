import distutils.sysconfig
from glob import glob
import os
from os.path import join as pjoin, basename
from shutil import copy
from sys import platform

def main():
    """
    Copy HDF5 DLLs into installed h5py package
    """
    # This is the function Tox also uses to locate site-packages (Apr 2019)
    sitepackagesdir = distutils.sysconfig.get_python_lib()
    print("site packages dir:", sitepackagesdir)

    hdf5_path = os.environ.get("HDF5_DIR")
    print("HDF5_DIR", hdf5_path)

    if platform.startswith('win'):
        for f in glob(pjoin(hdf5_path, 'lib/*.dll')):
            copy(f, pjoin(sitepackagesdir, 'h5py', basename(f)))
            print("Copied", f)

    print("In installed h5py:", os.listdir(pjoin(sitepackagesdir, 'h5py')))

if __name__ == '__main__':
    main()
