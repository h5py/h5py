import h5 as _h5

version = _h5._version_string
_exp = version.partition('-')
version_tuple = tuple(int(x) for x in _exp[0].split('.')) + (_exp[2],)


hdf5_version_tuple = _h5._hdf5_version_tuple
hdf5_version = "%d.%d.%d" % hdf5_version_tuple

api_version_tuple = _h5._api_version_tuple
api_version = "%d.%d" % api_version_tuple

__doc__ = """\
This is h5py **%s**

* HDF5 version: **%s**
* API compatibility: **%s**
""" % (version, hdf5_version, api_version)
