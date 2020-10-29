#!/bin/bash

set -e

if [ -z ${HDF5_DIR+x} ]; then
    echo "Using OS HDF5"
else
    echo "Using downloaded HDF5"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        lib_name=libhdf5.dylib
    else
        lib_name=libhdf5.so
    fi

    if [ -f $HDF5_DIR/lib/$lib_name ]; then
        echo "using cached build"
    else
        pushd /tmp
        #                                   Remove trailing .*, to get e.g. '1.12' ↓
        curl -fsSLO "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"
        tar -xzvf hdf5-$HDF5_VERSION.tar.gz
        pushd hdf5-$HDF5_VERSION
        chmod u+x autogen.sh
        if [[ "${HDF5_VERSION%.*}" = "1.12" ]]; then
          ./configure --prefix $HDF5_DIR --enable-build-mode=production
        else
          ./configure --prefix $HDF5_DIR
        fi
        make -j $(nproc)
        make install
        popd
        popd
    fi
fi
