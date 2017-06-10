#!/bin/sh

set -e

if [ -z ${HDF5_DIR+x} ]; then
    echo "Using OS HDF5"
else
    echo "Using downloaded HDF5"
    python3 -m pip install requests
    python3 ci/get_hdf5.py
fi
