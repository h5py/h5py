#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] || [[ "$2" == "" ]]; then
    echo "Usage: $0 <PROJECT_PATH> <WHEEL_PATH>"
    exit 1
fi
PROJECT_PATH=$1
echo "PROJECT_PATH=$PROJECT_PATH"
WHEEL_PATH=$2
echo "WHEEL_PATH=$WHEEL_PATH"

export PYVER=$(python -c "import sys; print(''.join(map(str, sys.version_info[:2])) + ('t' if (sys.version_info>=(3,13) and sys.implementation.name=='cpython' and not sys._is_gil_enabled()) else ''))")

if [[ "$PYVER" == *t ]]; then
    # TODO it was decided to not mark the module as freethreading-compatible
    # for the time being due to instability concerns.
    # See https://github.com/h5py/h5py/pull/2650
    export PYTHON_GIL="0"
fi

ENVLIST="py$PYVER-test-mindeps"
if [[ "$ONLY_MINDEPS_TESTS" != "1" ]] ; then
    ENVLIST="$ENVLIST,py$PYVER-test-deps,py$PYVER-test-deps-pre"
fi

export H5PY_TEST_CHECK_FILTERS=1
echo "ENVLIST=$ENVLIST"
cd $PROJECT_PATH
tox --installpkg $WHEEL_PATH -e $ENVLIST
if [[ "$GITHUB_ACTION" != "" ]]; then
    echo "Uploading coverage using python=$(which python)"
    python ./ci/upload_coverage.py --codecov-token 813fb6da-087d-4b36-a185-5a530cab3455
fi
