#!/bin/bash
set -e -u -x

# Installing hdf5 dependency
echo "Installing zlib with yum"
export HDF5_VERSION=1.12.0
export HDF5_DIR="/usr/local"
yum -y install zlib-devel

pushd /tmp
ldconfig

echo "Downloading & unpacking HDF5 ${HDF5_VERSION}"
curl -fsSLO "https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz"
tar -xzvf hdf5-$HDF5_VERSION.tar.gz
pushd hdf5-$HDF5_VERSION
chmod u+x autogen.sh

echo "Configuring, building & installing HDF5 ${HDF5_VERSION} to ${HDF5_DIR}"
./configure --prefix $HDF5_DIR --enable-build-mode=production --with-szlib
make -j $(nproc)
make install
popd

echo "Cleaning up unnecessary files"
rm -r hdf5-$HDF5_VERSION
rm hdf5-$HDF5_VERSION.tar.gz

yum -y erase zlib-devel

# Building wheels
cd /src/

# Create binary wheels
python=`ls /opt/python/cp*/bin/pip |grep "cp3[789]"`
for bindir in $python; do
    "$bindir" wheel /src/ -w wheelhouse/
done

ls -lrt wheelhouse/

# Normalize resulting binaries to a common format
for whl in wheelhouse/h5py-*.whl; do
    auditwheel repair "$whl" -w wheelhouse/
done
