# Install h5py in a convenient way for frequent reinstallation as you work on it.
# This disables the mechanisms to find and install build dependencies, so you
# need to already have those (Cython, pkgconfig, numpy & optionally mpi4py) installed
# in the current environment.
set -e

H5PY_SETUP_REQUIRES=0 python3 setup.py build
python3 -m pip install . --no-build-isolation
