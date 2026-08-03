#!/bin/bash
set -eo pipefail

if [[ -z "$1" ]]; then
  echo "Usage: $0 <Path to install zlib>"
  exit 1
fi

INSTALL_PREFIX="$1"

ZLIB_VERSION="1.3.2"
ZLIB_DIR="zlib-$ZLIB_VERSION"

if [ ! -d "$ZLIB_DIR" ]; then
  # zlib.net intermittently answers HTTP 200 with an error page, which
  # --fail cannot detect (it broke this job twice), so try the zlib GitHub
  # release mirror first and verify each download before accepting it.
  urls=(
      "https://github.com/madler/zlib/releases/download/v$ZLIB_VERSION/$ZLIB_DIR.tar.gz"
      "https://zlib.net/fossils/$ZLIB_DIR.tar.gz"
  )
  set +e
  for url in "${urls[@]}"; do
      echo "downloading from $url"
      curl --location "$url" --output "$ZLIB_DIR.tar.gz" --fail --silent --show-error --retry 5
      if [[ "$?" == 0 ]] && gzip -t "$ZLIB_DIR.tar.gz" 2>/dev/null; then
          echo "download succeeded"
          break
      else
          echo "download failed"
      fi
  done
  set -e
  gzip -t $ZLIB_DIR.tar.gz
  tar -xzf $ZLIB_DIR.tar.gz && rm $ZLIB_DIR.tar.gz
fi

cmake -S "$ZLIB_DIR" -B build \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
cmake --build build --config Release --parallel
cmake --install build --config Release
