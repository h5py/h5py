#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] ; then
    echo "Usage: $0 <PROJECT_PATH>"
    exit 1
fi
PROJECT_PATH="$1"
ARCH=$(uname -m)
export HDF5_VERSION="1.12.2"
export HDF5_DIR="$PROJECT_PATH/cache/hdf5/$HDF5_VERSION-$ARCH"
export MACOSX_DEPLOYMENT_TARGET="10.9"
source $PROJECT_PATH/ci/get_hdf5_if_needed.sh

if [[ "$GITHUB_ENV" != "" ]]; then
    echo "HDF5_DIR=$HDF5_DIR" >> $GITHUB_ENV
    echo "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET" >> $GITHUB_ENV
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $GITHUB_ENV
    echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH" >> $GITHUB_ENV
fi
