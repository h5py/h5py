#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] ; then
    echo "Usage: $0 <PROJECT_PATH>"
    exit 1
fi
PROJECT_PATH="$1"
ARCH=$(uname -m)
export HDF5_VERSION="1.14.6"
export HDF5_DIR="$PROJECT_PATH/cache/hdf5/$HDF5_VERSION-$ARCH"
source $PROJECT_PATH/ci/get_hdf5_if_needed.sh

if [[ "$GITHUB_ENV" != "" ]]; then
    echo "HDF5_DIR=$HDF5_DIR" | tee -a $GITHUB_ENV
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee -a $GITHUB_ENV
    echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH" | tee -a $GITHUB_ENV
fi
