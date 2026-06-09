# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Versioning module for h5py.
"""
from typing import Literal, NamedTuple
from . import h5 as _h5
import sys
import numpy

# All should be integers, except pre, as validating versions is more than is
# needed for our use case
class _H5PY_VERSION_CLS(NamedTuple):
    major: int
    minor: int
    bugfix: int
    pre: Literal["a", "b", "rc"] | None = None
    post: int | None = None
    dev: int | None = None

    def __str__(self) -> str:
        s = f"{self.major}.{self.minor}.{self.bugfix}"
        if self.pre is not None:
            s += version_tuple.pre
        if self.post is not None:
            s += f".post{self.post}"
        if version_tuple.dev is not None:
            s += f".dev{self.dev}"
        return s

hdf5_built_version_tuple = _h5.HDF5_VERSION_COMPILED_AGAINST

# keep in sync with project.version (pyproject.toml)
version_tuple = _H5PY_VERSION_CLS(major=3, minor=16, bugfix=0, dev=0)
version = str(version_tuple)

hdf5_version_tuple = _h5.get_libversion()
hdf5_version = "%d.%d.%d" % hdf5_version_tuple

api_version_tuple = (1,8)
api_version = "%d.%d" % api_version_tuple

info = """\
Summary of the h5py configuration
---------------------------------

h5py    %(h5py)s
HDF5    %(hdf5)s
Python  %(python)s
sys.platform    %(platform)s
sys.maxsize     %(maxsize)s
numpy   %(numpy)s
cython (built with) %(cython_version)s
numpy (built against) %(numpy_build_version)s
HDF5 (built against) %(hdf5_build_version)s
""" % {
    'h5py': version,
    'hdf5': hdf5_version,
    'python': sys.version,
    'platform': sys.platform,
    'maxsize': sys.maxsize,
    'numpy': numpy.__version__,
    'cython_version': _h5.CYTHON_VERSION_COMPILED_WITH,
    'numpy_build_version': _h5.NUMPY_VERSION_COMPILED_AGAINST,
    'hdf5_build_version': "%d.%d.%d" % hdf5_built_version_tuple,
}
