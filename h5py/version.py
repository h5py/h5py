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

from __future__ import absolute_import

from . import h5 as _h5
from distutils.version import StrictVersion as _sv
import sys
import numpy

version = "2.6.0"

_exp = _sv(version)

version_tuple = _exp.version + ((''.join(str(x) for x in _exp.prerelease),) if _exp.prerelease is not None else ('',))

hdf5_version_tuple = _h5.get_libversion()
hdf5_version = "%d.%d.%d" % hdf5_version_tuple

api_version_tuple = (1,8)
api_version = "1.8"

info = """\
Summary of the h5py configuration
---------------------------------

h5py    %(h5py)s
HDF5    %(hdf5)s
Python  %(python)s
sys.platform    %(platform)s
sys.maxsize     %(maxsize)s
numpy   %(numpy)s
""" % { 'h5py': version,
        'hdf5': hdf5_version,
        'python': sys.version,
        'platform': sys.platform,
        'maxsize': sys.maxsize,
        'numpy': numpy.__version__ }


