set -o errexit
rm -rf dist
python -m build
pip install dist/*.whl --force-reinstall --no-deps
mkdir -p /tmp/h5py
cp pytest.ini /tmp/h5py/
cd /tmp/h5py
# export PYTHON_GIL=0
python -m pytest --pyargs h5py -v --durations 10 --parallel-threads=32
