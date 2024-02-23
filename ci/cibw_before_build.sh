#!/bin/bash

set -eo pipefail

if ! python -c "import sys; exit(sys.version_info >= (3, 9))"; then
    NUMPY_ARGS="--pre --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy"
else
    NUMPY_ARGS="oldest-supported-numpy"
fi
pip install $NUMPY_ARGS "Cython>=0.29.31,<4" pkgconfig "setuptools>=61"
