#!/bin/bash

set -e -x

if [ -z ${HDF5_DIR+x} ]; then
    echo "Using OS HDF5"
else
    echo "Using downloaded HDF5"

    EXTRA_MPI_FLAGS=''
    if [ -z ${HDF5_MPI+x} ]; then
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
    fi

    if [ -f $HDF5_DIR/lib/$lib_name ]; then
        echo "using cached build"
    else
        echo "building HDF5"

        MINOR_V=${HDF5_VERSION#*.}
        MINOR_V=${MINOR_V%.*}
        MAJOR_V=${HDF5_VERSION/%.*.*}
        ZLIB_ARG=""

        if [[ "$OSTYPE" == "darwin"* ]]; then
            ZLIB_ARG="--with-zlib=$HDF5_DIR"
            export LD_LIBRARY_PATH="$HDF5_DIR/lib:${LD_LIBRARY_PATH}"
            export PKG_CONFIG_PATH="$HDF5_DIR/lib/pkgconfig:${PKG_CONFIG_PATH}"
            export CC="/usr/bin/clang"
            export CXX="/usr/bin/clang"
            export CFLAGS="$CFLAGS -arch $CIBW_ARCHS_MACOS"
            export CPPFLAGS="$CPPFLAGS -arch $CIBW_ARCHS_MACOS"
            export CXXFLAGS="$CXXFLAGS -arch $CIBW_ARCHS_MACOS"

            ZLIB_VERSION="1.2.11"
            GZIP_VERSION="1.11"

            pushd /tmp
            # zlib
            curl -sLO https://zlib.net/zlib-$ZLIB_VERSION.tar.gz
            tar xzf zlib-$ZLIB_VERSION.tar.gz
            cd zlib-$ZLIB_VERSION
            ./configure --prefix="$HDF5_DIR"
            make
            make install
            popd

            pushd /tmp
            # gzip
            curl -sLO https://ftp.sotirov-bg.net/pub/mirrors/gnu/gzip/gzip-$GZIP_VERSION.tar.xz
            tar xf gzip-$GZIP_VERSION.tar.xz
            cd gzip-$GZIP_VERSION
            ./configure --prefix="$HDF5_DIR"
            make
            make install
            popd
        fi

        pushd /tmp
        #                                   Remove trailing .*, to get e.g. '1.12' ↓
        curl -fsSLO "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"
        tar -xzvf hdf5-$HDF5_VERSION.tar.gz
        pushd hdf5-$HDF5_VERSION

        if [[ "$OSTYPE" == "darwin"* && "$CIBW_ARCHS_MACOS" = "arm64"  ]]; then
            # from https://github.com/conda-forge/hdf5-feedstock/commit/2cb83b63965985fa8795b0a13150bf0fd2525ebd
            export ac_cv_sizeof_long_double=8
            export hdf5_cv_ldouble_to_long_special=no
            export hdf5_cv_long_to_ldouble_special=no
            export hdf5_cv_ldouble_to_llong_accurate=yes
            export hdf5_cv_llong_to_ldouble_correct=yes
            export hdf5_cv_disable_some_ldouble_conv=no
            export hdf5_cv_system_scope_threads=yes
            export hdf5_cv_printf_ll="l"
            export PAC_FC_MAX_REAL_PRECISION=15
            export PAC_C_MAX_REAL_PRECISION=17
            export PAC_FC_ALL_INTEGER_KINDS="{1,2,4,8,16}"
            export PAC_FC_ALL_REAL_KINDS="{4,8}"
            export H5CONFIG_F_NUM_RKIND="INTEGER, PARAMETER :: num_rkinds = 2"
            export H5CONFIG_F_NUM_IKIND="INTEGER, PARAMETER :: num_ikinds = 5"
            export H5CONFIG_F_RKIND="INTEGER, DIMENSION(1:num_rkinds) :: rkind = (/4,8/)"
            export H5CONFIG_F_IKIND="INTEGER, DIMENSION(1:num_ikinds) :: ikind = (/1,2,4,8,16/)"
            export PAC_FORTRAN_NATIVE_INTEGER_SIZEOF="                    4"
            export PAC_FORTRAN_NATIVE_INTEGER_KIND="           4"
            export PAC_FORTRAN_NATIVE_REAL_SIZEOF="                    4"
            export PAC_FORTRAN_NATIVE_REAL_KIND="           4"
            export PAC_FORTRAN_NATIVE_DOUBLE_SIZEOF="                    8"
            export PAC_FORTRAN_NATIVE_DOUBLE_KIND="           8"
            export PAC_FORTRAN_NUM_INTEGER_KINDS="5"
            export PAC_FC_ALL_REAL_KINDS_SIZEOF="{4,8}"
            export PAC_FC_ALL_INTEGER_KINDS_SIZEOF="{1,2,4,8,16}"

            curl -sLO https://github.com/conda-forge/hdf5-feedstock/raw/2cb83b63965985fa8795b0a13150bf0fd2525ebd/recipe/patches/osx_cross_configure.patch
            curl -sLO https://github.com/conda-forge/hdf5-feedstock/raw/2cb83b63965985fa8795b0a13150bf0fd2525ebd/recipe/patches/osx_cross_fortran_src_makefile.patch
            curl -sLO https://github.com/conda-forge/hdf5-feedstock/raw/2cb83b63965985fa8795b0a13150bf0fd2525ebd/recipe/patches/osx_cross_hl_fortran_src_makefile.patch
            curl -sLO https://github.com/conda-forge/hdf5-feedstock/raw/2cb83b63965985fa8795b0a13150bf0fd2525ebd/recipe/patches/osx_cross_src_makefile.patch
            patch -p0 < osx_cross_configure.patch
            patch -p0 < osx_cross_fortran_src_makefile.patch
            patch -p0 < osx_cross_hl_fortran_src_makefile.patch
            patch -p0 < osx_cross_src_makefile.patch

            ./configure --prefix="$HDF5_DIR" $ZLIB_ARG "$EXTRA_MPI_FLAGS" --enable-build-mode=production \
                --host=aarch64-apple-darwin --enable-tests=no

            mkdir -p native-build/bin
            pushd native-build/bin
            CFLAGS= $CC ../../src/H5detect.c -I ../../src/ -o H5detect
            CFLAGS= $CC ../../src/H5make_libsettings.c -I ../../src/ -o H5make_libsettings
            popd
            export PATH=$(pwd)/native-build/bin:$PATH
        elif [[ $MAJOR_V -gt 1 || $MINOR_V -ge 12 ]]; then
            ./configure --prefix="$HDF5_DIR" $ZLIB_ARG $EXTRA_MPI_FLAGS --enable-build-mode=production --enable-tests=no
        else
            ./configure --prefix="$HDF5_DIR" $EXTRA_MPI_FLAGS --enable-tests=no
        fi

        make -j "$NPROC"
        make install
        popd
        popd
    fi
fi
