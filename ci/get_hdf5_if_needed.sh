#!/bin/bash

set -e -x

if [ -z ${HDF5_DIR+x} ]; then
    echo "Using OS HDF5"
else
    echo "Using downloaded HDF5"

    extra_arch_flags=()
    EXTRA_CMAKE_MPI_FLAGS=""
    EXTRA_MPI_FLAGS=''
    if [ -z ${HDF5_MPI+x} ]; then
        echo "Building serial"
    else
        echo "Building with MPI"
        EXTRA_CMAKE_MPI_FLAGS="-DHDF5_ENABLE_PARALLEL:bool=on"
        EXTRA_MPI_FLAGS="--enable-parallel --enable-shared"
    fi

    if [[ "$OSTYPE" == "darwin"* ]]; then
        lib_name=libhdf5.dylib
        NPROC=$(sysctl -n hw.ncpu)
    else
        lib_name=libhdf5.so
        NPROC=$(nproc)
    fi

    if [ -f $HDF5_DIR/lib/$lib_name ]; then
        echo "using cached build"
    else
        echo "building HDF5"

        MINOR_V=${HDF5_VERSION#*.}
        MINOR_V=${MINOR_V%.*}
        MAJOR_V=${HDF5_VERSION/%.*.*}
        if [[ "$OSTYPE" == "darwin"* && ( $MAJOR_V -gt 1 || $MINOR_V -ge 13 ) ]]; then
            brew install automake pkg-config cmake

            export LD_LIBRARY_PATH="$HDF5_DIR/lib:${LD_LIBRARY_PATH}"
            export PKG_CONFIG_PATH="$HDF5_DIR/lib/pkgconfig:${PKG_CONFIG_PATH}"
            export CC="/usr/bin/clang"
            export CXX="/usr/bin/clang"
            extra_arch_flags=("-DCMAKE_OSX_ARCHITECTURES='x86_64;arm64'")
            FLAGS="-arch x86_64 -arch arm64"
            ZLIB_VERSION="1.2.11"
            GZIP_VERSION="1.11"

            pushd /tmp
            # zlib
            curl -sLO https://zlib.net/zlib-$ZLIB_VERSION.tar.gz
            tar xzf zlib-$ZLIB_VERSION.tar.gz
            cd zlib-$ZLIB_VERSION
            CFLAGS="$CFLAGS $FLAGS" ./configure --prefix="$HDF5_DIR"
            CFLAGS="$CFLAGS $FLAGS" make
            make install
            popd

            pushd /tmp
            # gzip
            curl -sLO https://ftp.sotirov-bg.net/pub/mirrors/gnu/gzip/gzip-$GZIP_VERSION.tar.xz
            tar xf gzip-$GZIP_VERSION.tar.xz
            cd gzip-$GZIP_VERSION
            CFLAGS="$CFLAGS $FLAGS" ./configure --prefix="$HDF5_DIR"
            CFLAGS="$CFLAGS $FLAGS" make
            make install
            popd
        fi

        pushd /tmp
        #                                   Remove trailing .*, to get e.g. '1.12' ↓
        curl -fsSLO "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"
        tar -xzvf hdf5-$HDF5_VERSION.tar.gz
        pushd hdf5-$HDF5_VERSION

        if [[ $MAJOR_V -gt 1 || $MINOR_V -ge 13 ]]; then
            mkdir build
            cd build
            cmake -DCMAKE_INSTALL_PREFIX="$HDF5_DIR" -DENABLE_SHARED:bool=on $EXTRA_CMAKE_MPI_FLAGS "${extra_arch_flags[@]}" ../
        elif [ $MINOR_V -eq 12 ]; then
            ./configure --prefix "$HDF5_DIR" $EXTRA_MPI_FLAGS --enable-build-mode=production
        else
            ./configure --prefix "$HDF5_DIR" $EXTRA_MPI_FLAGS
        fi
        make -j "$NPROC"
        make install
        popd
        popd
    fi
fi
