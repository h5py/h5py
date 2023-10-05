import os
import sysconfig
from glob import glob
from os.path import join as pjoin, basename
from shutil import copy
from sys import platform

def main():
    """
    Copy HDF5 DLLs into installed h5py package
    """
    sitepackagesdir = sysconfig.get_path('platlib')
    print("site packages dir:", sitepackagesdir)

    hdf5_path = os.environ.get("HDF5_DIR")
    print("HDF5_DIR", hdf5_path)

    # HDF5_DIR is not set when we're testing wheels; these should already have
    # the necessary libraries bundled in.
    if platform.startswith('win') and hdf5_path is not None:
        for f in glob(pjoin(hdf5_path, 'lib/*.dll')):
            copy(f, pjoin(sitepackagesdir, 'h5py', basename(f)))
            print("Copied", f)

        zlib_root = os.environ.get("ZLIB_ROOT")
        if zlib_root:
            f = pjoin(zlib_root, 'bin_release', 'zlib.dll')
            copy(f, pjoin(sitepackagesdir, 'h5py', 'zlib.dll'))
            print("Copied", f)

        print("In installed h5py:", sorted(os.listdir(pjoin(sitepackagesdir, 'h5py'))))

if __name__ == '__main__':
    main()
