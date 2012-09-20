from . import h5 as _h5
from distutils.version import StrictVersion as _sv

version = "2.2.0a1"
_exp = _sv(version)

version_tuple = _exp.version + (''.join(str(x) for x in _exp.prerelease),)

hdf5_version_tuple = _h5.get_libversion()
hdf5_version = "%d.%d.%d" % hdf5_version_tuple

api_version_tuple = (1,8)
api_version = "1.8"

__doc__ = """\
This is h5py **%s**

* HDF5 version: **%s**
* API compatibility: **%s**
""" % (version, hdf5_version, api_version)
