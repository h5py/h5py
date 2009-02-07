#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

# h5py module __init__

__doc__ = \
"""
    This is the h5py package, a Python interface to the HDF5 
    scientific data format.

    Version %s

    HDF5 %s (using %s API)
"""
try:
    import h5
except ImportError, e:
    import os.path as op
    if op.exists('setup.py'):
        raise ImportError('Import error:\n"%s"\n\nBe sure to exit source directory before importing h5py' % e)
    raise

import utils, h5, h5a, h5d, h5f, h5fd, h5g, h5i, h5p, h5r, h5s, h5t, h5z, highlevel, version

from highlevel import File, Group, Dataset, Datatype, AttributeManager, is_hdf5, CoordsList
from h5 import H5Error, get_config

import filters, selections

__doc__ = __doc__ % (version.version, version.hdf5_version, version.api_version)

__all__ = ['h5', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p', 'h5r',
           'h5z', 'h5i', 'version', 'File', 'Group', 'Dataset',
           'Datatype', 'AttributeManager', 'H5Error', 'get_config', 'is_hdf5']

if version.api_version_tuple >= (1,8):
    import h5o, h5l
    __all__ += ['h5l', 'h5o']

try:
   import IPython as _IP
   if _IP.ipapi.get() is not None:
       import _ipy_completer
       _ipy_completer.activate()
except Exception:
   pass

