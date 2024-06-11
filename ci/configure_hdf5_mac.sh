function build_zlib() {
    ZLIB_VERSION="1.2.13"

    pushd /tmp
    curl -sLO https://zlib.net/fossils/zlib-$ZLIB_VERSION.tar.gz
    tar xzf zlib-$ZLIB_VERSION.tar.gz
    cd zlib-$ZLIB_VERSION
    ./configure --prefix="$HDF5_DIR"
    make
    make install
    popd
}
