function set_compiler_vars() {
    arch=$1
    export CC="/usr/bin/clang"
    export CXX="/usr/bin/clang"
    export CFLAGS="$CFLAGS -arch $arch"
    export CPPFLAGS="$CPPFLAGS -arch $arch"
    export CXXFLAGS="$CXXFLAGS -arch $arch"
}


function build_zlib() {
    ZLIB_VERSION="1.2.12"
    # export needed to fix 1.2.12. Next release won't need the export (madler/zlib#615)
    export cc=$CC

    pushd /tmp
    curl -sLO https://zlib.net/fossils/zlib-$ZLIB_VERSION.tar.gz
    tar xzf zlib-$ZLIB_VERSION.tar.gz
    cd zlib-$ZLIB_VERSION
    ./configure --prefix="$HDF5_DIR"
    make
    make install
    popd
}


function patch_hdf5() {
    # from https://github.com/conda-forge/hdf5-feedstock/commit/2cb83b63965985fa8795b0a13150bf0fd2525ebd
    export ac_cv_sizeof_long_double=8
    export hdf5_cv_ldouble_to_long_special=no
    export hdf5_cv_long_to_ldouble_special=no
    export hdf5_cv_ldouble_to_llong_accurate=yes
    export hdf5_cv_llong_to_ldouble_correct=yes
    export hdf5_cv_disable_some_ldouble_conv=no
    export hdf5_cv_system_scope_threads=yes
    export hdf5_cv_printf_ll="l"

    patch -p0 < "$1/ci/osx_cross_configure.patch"
    patch -p0 < "$1/ci/osx_cross_src_makefile.patch"
}


function build_h5detect() {
    mkdir -p native-build/bin
    pushd native-build/bin
    CFLAGS= $CC ../../src/H5detect.c -I ../../src/ -o H5detect
    CFLAGS= $CC ../../src/H5make_libsettings.c -I ../../src/ -o H5make_libsettings
    popd

    export PATH="$(pwd)/native-build/bin:$PATH"
}
