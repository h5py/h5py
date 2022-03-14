#!/bin/bash

set -e -x

if [ -z ${HDF5_DIR+x} ]; then
    echo "Using OS HDF5"
else
    echo "Using downloaded HDF5"
    if [ -z ${HDF5_MPI+x} ]; then
        echo "Building serial"
        EXTRA_MPI_FLAGS=''
    else
        echo "Building with MPI"
        EXTRA_MPI_FLAGS="--enable-parallel --enable-shared"
    fi

    if [[ "$OSTYPE" == "darwin"* ]]; then
        lib_name=libhdf5.dylib
    else
        lib_name=libhdf5.so
    fi

    if [ -f $HDF5_DIR/lib/$lib_name ]; then
        echo "using cached build"
    else
        echo "building HDF5"

        MINOR_V=${HDF5_VERSION#*.}
        MINOR_V=${MINOR_V%.*}
        MAJOR_V=${HDF5_VERSION/%.*.*}
        if [[ "$OSTYPE" == "darwin"* && ( $MAJOR_V -gt 1 || $MINOR_V -ge 13 ) ]]; then
            brew install automake pkg-config

            export LD_LIBRARY_PATH="$HDF5_DIR/lib:${LD_LIBRARY_PATH}"
            export PKG_CONFIG_PATH="$HDF5_DIR/lib/pkgconfig:${PKG_CONFIG_PATH}"
            export CFLAGS="$CFLAGS -arch x86_64 -arch arm64"
            export CC="/usr/bin/clang"
            export CXX="/usr/bin/clang"
            ZLIB_VERSION="1.2.11"

            pushd /tmp
            # zlib
            curl -sLO https://zlib.net/zlib-$ZLIB_VERSION.tar.gz
            tar xzf zlib-$ZLIB_VERSION.tar.gz
            cd zlib-$ZLIB_VERSION
            CPPFLAGS="$CPPFLAGS -arch x86_64 -arch arm64" CXXFLAGS="$CXXFLAGS -arch x86_64 -arch arm64" ./configure --prefix="$HDF5_DIR"
            CPPFLAGS="$CPPFLAGS -arch x86_64 -arch arm64" CXXFLAGS="$CXXFLAGS -arch x86_64 -arch arm64" make
            make install
            popd
        fi

        pushd /tmp
        #                                   Remove trailing .*, to get e.g. '1.12' ↓
        curl -fsSLO "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"
        tar -xzvf hdf5-$HDF5_VERSION.tar.gz
        pushd hdf5-$HDF5_VERSION
        chmod u+x autogen.sh
        if [[ $MAJOR_V -gt 1 || $MINOR_V -ge 12 ]]; then
          ./configure --prefix $HDF5_DIR $EXTRA_MPI_FLAGS --enable-build-mode=production
        else
          ./configure --prefix $HDF5_DIR $EXTRA_MPI_FLAGS
        fi
        make -j $(nproc)
        make install
        popd
        popd
    fi
fi
