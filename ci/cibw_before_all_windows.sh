#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] ; then
    echo "Usage: $0 <PROJECT_PATH>"
    exit 1
fi
PROJECT_PATH="$1"

if [[ "$ARCH" == "ARM64" ]]; then
    # Build zlib from source
    ./ci/get_zlib_amd64.sh $PROJECT_PATH/zlib-win-arm64

    ZLIB_ROOT="$PROJECT_PATH/zlib-win-arm64"
    EXTRA_PATH="$ZLIB_ROOT/bin"
    export CL="/I$ZLIB_ROOT/include"
    export LINK="/LIBPATH:$ZLIB_ROOT/lib"
    export HDF5_VSVERSION="17-arm64"
elif [[ "$ARCH" == "AMD64" ]]; then
    # Build zlib from source for Windows AMD64
    ./ci/get_zlib_amd64.sh $PROJECT_PATH/zlib-win-x64

    ZLIB_ROOT="$PROJECT_PATH/zlib-win-x64"
    EXTRA_PATH="$ZLIB_ROOT/bin"
    export CL="/I$ZLIB_ROOT/include"
    export LINK="/LIBPATH:$ZLIB_ROOT/lib"
    export HDF5_VSVERSION="17-64"
else
    echo "Got unexpected arch '$ARCH'"
    exit 1
fi

export PATH="$PATH:$EXTRA_PATH"
export ZLIB_ROOT

# HDF5
export HDF5_VERSION="2.0.0"
export HDF5_DIR="$PROJECT_PATH/cache/hdf5/$HDF5_VERSION"

pip install requests
python $PROJECT_PATH/ci/get_hdf5_win.py

if [[ "$GITHUB_ENV" != "" ]] ; then
    # PATH on windows is special
    echo "$EXTRA_PATH" | tee -a $GITHUB_PATH
    echo "CL=$CL" | tee -a $GITHUB_ENV
    echo "LINK=$LINK" | tee -a $GITHUB_ENV
    echo "ZLIB_ROOT=$ZLIB_ROOT" | tee -a $GITHUB_ENV
    echo "HDF5_DIR=$HDF5_DIR" | tee -a $GITHUB_ENV
fi
