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

export PYVER=$(python -c "import sys; print(''.join(map(str, sys.version_info[:2])))")
ENVLIST="py$PYVER-test-deps,py$PYVER-test-deps-pre"

if ! ( [[ "$AUDITWHEEL_PLAT" == musllinux_* ]] && [[ "$PYVER" == "39" || "$PYVER" == "310" || "$PYVER" == "311" ]] ); then
    # skip mindeps on musllinux + python 3.9 to 3.11 because oldest supported numpy versions
    # are higher for this target (1.25.0) and there's no way (that I found) to specify it
    # directly in package metadata.
    ENVLIST="py$PYVER-test-mindeps,$ENVLIST"
fi

export H5PY_TEST_CHECK_FILTERS=1
echo "ENVLIST=$ENVLIST"
cd $PROJECT_PATH
tox --installpkg $WHEEL_PATH -e $ENVLIST
if [[ "$GITHUB_ACTION" != "" ]]; then
    echo "Uploading coverage using python=$(which python)"
    python ./ci/upload_coverage.py --codecov-token 813fb6da-087d-4b36-a185-5a530cab3455
fi
