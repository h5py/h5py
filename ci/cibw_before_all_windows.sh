#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] ; then
    echo "Usage: $0 <PROJECT_PATH>"
    exit 1
fi
PROJECT_PATH="$1"

# nuget
nuget install zlib-msvc-x64 -ExcludeVersion -OutputDirectory "$PROJECT_PATH"
EXTRA_PATH="$PROJECT_PATH\zlib-msvc-x64\build\native\bin_release"
export PATH="$PATH:$EXTRA_PATH"
export CL="/I$PROJECT_PATH\zlib-msvc-x64\build\native\include"
export LINK="/LIBPATH:$PROJECT_PATH\zlib-msvc-x64\build\native\lib_release"
export ZLIB_ROOT="$PROJECT_PATH\zlib-msvc-x64\build\native"

# HDF5
export HDF5_VERSION="1.14.2"
export HDF5_VSVERSION="17-64"
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
