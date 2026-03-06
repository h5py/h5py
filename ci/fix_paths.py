import os
import sysconfig
import platform
from glob import glob
from os.path import join as pjoin, basename
from shutil import copy
from sys import platform as sys_platform

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
    if sys_platform.startswith('win') and hdf5_path is not None:
        for f in glob(pjoin(hdf5_path, 'lib/*.dll')):
            copy(f, pjoin(sitepackagesdir, 'h5py', basename(f)))
            print("Copied", f)

        zlib_root = os.environ.get("ZLIB_ROOT")
        if zlib_root:
            arch = platform.machine().lower()
            # z.dll from zlib 1.3.2; previously zlib.dll / zlib1.dll
            f = pjoin(zlib_root, "bin", "z.dll")
            copy(f, pjoin(sitepackagesdir, 'h5py', basename(f)))
            print("Copied", f)

        print("In installed h5py:", sorted(os.listdir(pjoin(sitepackagesdir, 'h5py'))))

if __name__ == '__main__':
    main()
