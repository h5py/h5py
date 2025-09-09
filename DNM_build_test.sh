#!/bin/bash
set -o errexit

eval "$(mamba shell hook --shell bash)"
mamba create -y -n h5py-314t python-freethreading=3.14 numpy hdf5 python-build pytest pytest-mpi pytest-run-parallel -c conda-forge -c conda-forge/label/python_rc
mamba activate h5py-314t

rm -rf dist
python -m build
pip install dist/*.whl --force-reinstall --no-deps
mkdir -p /tmp/h5py
cp pytest.ini /tmp/h5py/
cd /tmp/h5py
export PYTHON_GIL=0
python -m pytest --pyargs h5py -v --durations 10 --parallel-threads=32
