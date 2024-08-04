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
# When compiling HDF5, we should use the minimum across all Python versions for a given
# arch, for versions see for example a more updated version of the following:
# https://github.com/pypa/cibuildwheel/blob/89a5cfe2721c179f4368a2790669e697759b6644/cibuildwheel/macos.py#L296-L310
if [[ "$ARCH" == "arm64" ]]; then
    export MACOSX_DEPLOYMENT_TARGET="11.0"
else
    export MACOSX_DEPLOYMENT_TARGET="10.9"
fi
source $PROJECT_PATH/ci/get_hdf5_if_needed.sh

if [[ "$GITHUB_ENV" != "" ]]; then
    echo "HDF5_DIR=$HDF5_DIR" | tee -a $GITHUB_ENV
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee -a $GITHUB_ENV
    echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH" | tee -a $GITHUB_ENV
fi
