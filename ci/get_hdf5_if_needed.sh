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
            export LD_LIBRARY_PATH="$HDF5_DIR/lib:${LD_LIBRARY_PATH}"
            export PKG_CONFIG_PATH="$HDF5_DIR/lib/pkgconfig:${PKG_CONFIG_PATH}"

            LZ4_VERSION="1.9.3"
            BZIP_VERSION="1.0.8"
            ZLIB_VERSION="1.2.11"

            brew install automake cmake pkg-config

            pushd /tmp

            export CFLAGS="$CFLAGS -arch x86_64 -arch arm64"
            export CPPFLAGS="$CPPFLAGS -arch x86_64 -arch arm64"
            export CXXFLAGS="$CXXFLAGS -arch x86_64 -arch arm64"
            export CC="/usr/bin/clang"
            export CXX="/usr/bin/clang"

            # lz4
            curl -sLO https://github.com/lz4/lz4/archive/refs/tags/v$LZ4_VERSION.tar.gz
            tar xzf v$LZ4_VERSION.tar.gz
            pushd lz4-$LZ4_VERSION
            make install PREFIX="$HDF5_DIR"
            popd

            # bzip2
            curl -sLO https://gitlab.com/bzip2/bzip2/-/archive/bzip2-$BZIP_VERSION/bzip2-bzip2-$BZIP_VERSION.tar.gz
            tar xzf bzip2-bzip2-$BZIP_VERSION.tar.gz
            pushd bzip2-bzip2-$BZIP_VERSION
            cat << EOF >> Makefile

libbz2.dylib: \$(OBJS)
	\$(CC) \$(LDFLAGS) -shared -Wl,-install_name -Wl,libbz2.dylib -o libbz2.$BZIP_VERSION.dylib \$(OBJS)
	cp libbz2.$BZIP_VERSION.dylib \${PREFIX}/lib/
	ln -s libbz2.$BZIP_VERSION.dylib \${PREFIX}/lib/libbz2.1.0.dylib
	ln -s libbz2.$BZIP_VERSION.dylib \${PREFIX}/lib/libbz2.dylib

EOF
            sed -i "" "s/CFLAGS=-Wall/CFLAGS=-fPIC -Wall/g" Makefile
            sed -i "" "s/all: libbz2.a/all: libbz2.dylib libbz2.a/g" Makefile
            cat Makefile
            make install PREFIX="$HDF5_DIR"
            popd

            # zlib
            curl -sLO https://zlib.net/zlib-$ZLIB_VERSION.tar.gz
            tar xzf zlib-$ZLIB_VERSION.tar.gz
            pushd zlib-$ZLIB_VERSION
            ./configure --prefix="$HDF5_DIR"
            make
            make install
            popd

            popd

            export CPPFLAGS=
            export CXXFLAGS=
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
