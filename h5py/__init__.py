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
    # Many people try to load h5py after compiling, which fails in the
    # presence of the source directory
    import os.path as op
    if op.exists('setup.py'):
        raise ImportError('Import error:\n"%s"\n\nBe sure to exit source directory before importing h5py' % e)
    raise

# This is messy but much less frustrating when using h5py from IPython
import h5, h5a, h5d, h5f, h5fd, h5g, h5l, h5o, h5i, h5p, h5r, h5s, h5t, h5z
import highlevel, filters, selections, version

from h5 import get_config
from h5e import H5Error

from highlevel import File, Group, Dataset, Datatype, AttributeManager, \
                      SoftLink, ExternalLink, is_hdf5

# New way to handle special types
from h5t import special_dtype, check_dtype
from h5r import Reference, RegionReference

# Deprecated way to handle special types
# These are going away in 1.4
from h5t import py_new_vlen as new_vlen
from h5t import py_get_vlen as get_vlen
from h5t import py_new_enum as new_enum
from h5t import py_get_enum as get_enum

__doc__ = __doc__ % (version.version, version.hdf5_version, version.api_version)

__all__ = ['h5', 'h5f', 'h5g', 'h5s', 'h5t', 'h5d', 'h5a', 'h5p', 'h5r',
           'h5o', 'h5l', 'h5z', 'h5i', 'version', 'File', 'Group', 'Dataset',
           'Datatype', 'AttributeManager', 'H5Error', 'get_config', 'is_hdf5',
           'special_dtype', 'check_dtype', 'SoftLink', 'ExternalLink']

try:
    try:
        import IPython.core.ipapi as _IP
    except ImportError:
        # support <ipython-0.11
        import IPython.ipapi as _IP
    if _IP.get() is not None:
        import _ipy_completer
        _ipy_completer.activate()
except Exception:
    pass


