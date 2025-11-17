#!/bin/bash
set -eo pipefail

if [[ -z "$1" ]]; then
  echo "Usage: $0 <Path to install zlib>"
  exit 1
fi

INSTALL_PREFIX="$1"

ZLIB_VERSION="1.3.1"
ZLIB_DIR="zlib-$ZLIB_VERSION"

if [ ! -d "$ZLIB_DIR" ]; then
  curl -sLO https://zlib.net/fossils/$ZLIB_DIR.tar.gz
  tar -xzf $ZLIB_DIR.tar.gz && rm $ZLIB_DIR.tar.gz
fi

cmake -S "$ZLIB_DIR" -B build -G "Visual Studio 17 2022" \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
cmake --build build --config Release --parallel
cmake --install build --config Release
