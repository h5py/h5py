#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] || [[ "$2" == "" ]]; then
    echo "Usage: $0 <ARCH> <SHA> <PYTHON>"
    exit 1
fi

ARCH=$1
SHA=$2
PYTHON=$3
if [[ "$PYTHON" == "" ]]; then
    PYTHON="*"
fi
MSG="$(git show -s --format=%s $SHA)"
KIND="$RUNNER_OS $ARCH"

CIBW_SKIP="pp* *musllinux*"
# If it's a scheduled build or [pip-pre] in commit message, use pip-pre
if [[ "$GITHUB_EVENT_NAME" == "schedule" ]] || [[ "$MSG" = *'[pip-pre]'* ]]; then
    echo "Using NumPy pip-pre wheel and (on Linux), setting CIBW_BEFORE_BUILD, CIBW_BUILD_FRONTEND and CIBW_BEFORE_TEST"
    echo "CIBW_BEFORE_BUILD=pip install --pre --only-binary numpy --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \"numpy>=2.0.0.dev0\" \"Cython>=0.29.31,<4\" pkgconfig \"setuptools>=61\" wheel" | tee -a $GITHUB_ENV
    echo "CIBW_BUILD_FRONTEND=pip; args: --no-build-isolation" | tee -a $GITHUB_ENV
    # This is harder on other architectures, so only do it on Linux for now
    echo "CIBW_BEFORE_TEST=pip install --pre --only-binary numpy --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \"numpy>=2.0.0.dev0\"" | tee -a $GITHUB_ENV
fi

# strip '-dev' suffix for pre-releases Pythons
PYTHON="${PYTHON%-dev*}"

# replace dots in PYTHON with nothing, e.g., 3.8->38
CIBW_BUILD="cp${PYTHON//./}-*_$ARCH"
echo "CIBW_BUILD=$CIBW_BUILD" | tee -a $GITHUB_ENV
echo "CIBW_SKIP=$CIBW_SKIP" | tee -a $GITHUB_ENV
echo "CIBW_PRERELEASE_PYTHONS=$CIBW_PRERELEASE_PYTHONS" | tee -a $GITHUB_ENV
