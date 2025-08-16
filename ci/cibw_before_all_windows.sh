#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] ; then
    echo "Usage: $0 <PROJECT_PATH>"
    exit 1
fi
PROJECT_PATH="$1"

if [[ "$ARCH" == "ARM64" ]]; then
    #Use vcpkg for Windows ARM64, since Nuget\Chocolatey doesn't provide zlib package for Windows ARM64
    VCPKG_ROOT="$PROJECT_PATH/vcpkg"
    VCPKG_SHA="dd3097e305afa53f7b4312371f62058d2e665320"  # 2025.07.25
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
    (cd "$VCPKG_ROOT" && git checkout "$VCPKG_SHA")
    $VCPKG_ROOT/bootstrap-vcpkg.bat -disableMetrics
    VCPKG_TRIPLET="arm64-windows"
    $VCPKG_ROOT/vcpkg.exe install zlib:$VCPKG_TRIPLET
    ZLIB_ROOT="$VCPKG_ROOT/installed/$VCPKG_TRIPLET"
    EXTRA_PATH="$ZLIB_ROOT/bin"
    export CL="/I$ZLIB_ROOT/include"
    export LINK="/LIBPATH:$ZLIB_ROOT/lib"
    export HDF5_VSVERSION="17-arm64"
elif [[ "$ARCH" == "x86_64" ]]; then
    # NuGet for Windows x64
    nuget install zlib-msvc-x64 -ExcludeVersion -OutputDirectory "$PROJECT_PATH"
    ZLIB_ROOT="$PROJECT_PATH/zlib-msvc-x64/build/native"
    EXTRA_PATH="$ZLIB_ROOT/bin_release"
    export CL="/I$ZLIB_ROOT/include"
    export LINK="/LIBPATH:$ZLIB_ROOT/lib_release"
    export HDF5_VSVERSION="17-64"
else
    echo "Got unexpected arch '$ARCH'"
    exit 1
fi

export PATH="$PATH:$EXTRA_PATH"
export ZLIB_ROOT

# HDF5
export HDF5_VERSION="1.14.6"
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
