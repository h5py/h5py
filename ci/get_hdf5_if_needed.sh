#!/bin/bash

set -e -x

PROJECT_ROOT="$(pwd)"
source "$PROJECT_ROOT/ci/configure_hdf5_mac.sh"

if [ -z ${HDF5_DIR+x} ]; then
    echo "Using OS HDF5"
else
    echo "Using downloaded HDF5"

    if [[ ${HDF5_MPI} != "ON" ]]; then
        echo "Building serial"
    else
        echo "Building with MPI"
        EXTRA_MPI_FLAGS="--enable-parallel --enable-shared"
    fi

    if [[ "$OSTYPE" == "darwin"* ]]; then
        lib_name=libhdf5.dylib
        NPROC=$(sysctl -n hw.ncpu)
    else
        lib_name=libhdf5.so
        NPROC=$(nproc)
        # Test with the direct file driver on Linux. This setting does not
        # affect the HDF5 bundled in Linux wheels - that is built into a Docker
        # image from a separate repository.
        ENABLE_DIRECT_VFD="--enable-direct-vfd"
    fi

    if [ -f $HDF5_DIR/lib/$lib_name ]; then
        echo "using cached build"
    else
        echo "building HDF5"

        MINOR_V=${HDF5_VERSION#*.}
        MINOR_V=${MINOR_V%.*}
        MAJOR_V=${HDF5_VERSION/%.*.*}
        if [[ $MAJOR_V -gt 1 || $MINOR_V -ge 12 ]]; then
            BUILD_MODE="--enable-build-mode=production"
        fi

        if [[ "$OSTYPE" == "darwin"* ]]; then
            set_compiler_vars "$CIBW_ARCHS_MACOS"
            build_zlib

            export LD_LIBRARY_PATH="$HDF5_DIR/lib:${LD_LIBRARY_PATH}"
            export PKG_CONFIG_PATH="$HDF5_DIR/lib/pkgconfig:${PKG_CONFIG_PATH}"
            ZLIB_ARG="--with-zlib=$HDF5_DIR"
            if [[ "$CIBW_ARCHS_MACOS" = "arm64"  ]]; then
                HOST_ARG="--host=aarch64-apple-darwin"
            fi
        fi

        pushd /tmp
        #                                   Remove trailing .*, to get e.g. '1.12' ↓
        curl -fsSLO "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-$MAJOR_V.$MINOR_V/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"
        tar -xzvf hdf5-$HDF5_VERSION.tar.gz
        pushd hdf5-$HDF5_VERSION

        if [[ "$OSTYPE" == "darwin"* && "$CIBW_ARCHS_MACOS" = "arm64"  ]]; then
            patch_hdf5 "$PROJECT_ROOT"
        fi

        ./configure --prefix="$HDF5_DIR" \
            ${ZLIB_ARG} \
            ${EXTRA_MPI_FLAGS} \
            ${BUILD_MODE} \
            ${ENABLE_DIRECT_VFD} \
            ${HOST_ARG} \
            --enable-tests=no

        if [[ "$OSTYPE" == "darwin"* && "$CIBW_ARCHS_MACOS" = "arm64"  ]]; then
            build_h5detect
        fi

        make -j "$NPROC"
        make install
        popd
        popd

        file "$HDF5_DIR"/lib/*
    fi
fi
