from . import h5 as _h5

version = "2.0.1"
_exp = version.partition('-')
version_tuple = tuple(int(x) for x in _exp[0].split('.')) + (_exp[2],)

hdf5_version_tuple = _h5.get_libversion()
hdf5_version = "%d.%d.%d" % hdf5_version_tuple

api_version_tuple = (1,8)
api_version = "1.8"

__doc__ = """\
This is h5py **%s**

* HDF5 version: **%s**
* API compatibility: **%s**
""" % (version, hdf5_version, api_version)
