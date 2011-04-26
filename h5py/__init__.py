import _errors
_errors.silence_errors()

import _conv
_conv.register_converters()

import h5a, h5d, h5f, h5fd, h5g, h5r, h5s, h5t, h5p, h5z

h5z._register_lzf()

from highlevel import *

from h5 import get_config
from h5r import Reference, RegionReference
from h5t import special_dtype, check_dtype

# Deprecated functions
from h5t import py_new_vlen as new_vlen
from h5t import py_get_vlen as get_vlen
from h5t import py_new_enum as new_enum
from h5t import py_get_enum as get_enum

import version

__doc__ = \
"""
    This is the h5py package, a Python interface to the HDF5 
    scientific data format.

    Version %s

    HDF5 %s
""" % (version.version, version.hdf5_version)


import sys as _sys
if 'IPython' in _sys.modules:
    _ip_running = False
    try:
        from IPython.core.iplib import InteractiveShell as _ishell
        _ip_running = _ishell.initialized()
    except ImportError:
        # support <ipython-0.11
        from IPython import ipapi as _ipapi
        _ip_running = _ipapi.get() is not None
    except Exception:
        pass
    if _ip_running:
        import _ipy_completer
        _ipy_completer.activate()

