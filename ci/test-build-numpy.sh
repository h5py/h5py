#!/bin/bash

set -e -x

git clone https://github.com/numpy/numpy.git
git checkout v1.24.3
cd numpy
python3 setup.py build
