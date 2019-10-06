#!/bin/bash

set -e

if [ -z ${HDF5_DIR+x} ]; then
    echo "No HDF5_DIR set, exiting"
    exit 1
else
    echo "Using development HDF5"
    pushd /tmp
    git clone https://bitbucket.hdfgroup.org/scm/hdffv/hdf5.git hdf5
    pushd hdf5
    git checkout ${HDF5_BRANCH:-develop}
    chmod u+x autogen.sh
    ./autogen.sh
    ./configure --prefix $HDF5_DIR $HDF5_CONFIG_ARGS
    make -j $(nproc)
    make install
    popd
    popd
fi
