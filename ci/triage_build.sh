#!/bin/bash

# REMEMBER TO KEEP THE VERSIONS IN THIS FILE THE SAME AS IN THE pyproject.toml

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

# If it's a scheduled build or [pip-pre] in commit message, use pip-pre
if [[ "$GITHUB_EVENT_NAME" == "schedule" ]] || [[ "$MSG" = *'[pip-pre]'* ]]; then
    echo "Using NumPy pip-pre wheel; setting CIBW_BEFORE_BUILD and CIBW_BUILD_FRONTEND"
    echo "CIBW_BEFORE_BUILD=pip install --pre --only-binary numpy --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \"numpy>=2.0.0.dev0\" \"Cython>=0.29.31,<4\" pkgconfig \"setuptools>=77\"" | tee -a $GITHUB_ENV
    echo "CIBW_BUILD_FRONTEND=pip; args: --no-build-isolation" | tee -a $GITHUB_ENV
fi

# strip '-dev' suffix for pre-releases Pythons
PYTHON="${PYTHON%-dev*}"

# replace dots in PYTHON with nothing, e.g., 3.8->38
CIBW_BUILD="cp${PYTHON//./}-*_$ARCH"
echo "CIBW_BUILD=$CIBW_BUILD" | tee -a $GITHUB_ENV
