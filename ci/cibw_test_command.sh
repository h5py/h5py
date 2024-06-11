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
export TOXENV="py$PYVER-test-deps,py$PYVER-test-mindeps,py$PYVER-test-deps-pre"
export H5PY_TEST_CHECK_FILTERS=1
echo "TOXENV=$TOXENV"
tox --installpkg $WHEEL_PATH
if [[ "$GITHUB_ACTION" != "" ]]; then
    echo "Uploading coverage"
    python $PROJECT_PATH/ci/upload_coverage.py --codecov-token 813fb6da-087d-4b36-a185-5a530cab3455
fi
