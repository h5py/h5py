# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    This is the h5py package, a Python interface to the HDF5
    scientific data format.
"""

from __future__ import absolute_import

from warnings import warn as _warn


# --- Library setup -----------------------------------------------------------

# When importing from the root of the unpacked tarball or git checkout,
# Python sees the "h5py" source directory and tries to load it, which fails.
# We tried working around this by using "package_dir" but that breaks Cython.
try:
    from . import _errors
except ImportError:
    import os.path as _op
    if _op.exists(_op.join(_op.dirname(__file__), '..', 'setup.py')):
        raise ImportError("You cannot import h5py from inside the install directory.\nChange to another directory first.")
    else:
        raise

_errors.silence_errors()

from ._conv import register_converters as _register_converters
_register_converters()

from .h5z import _register_lzf
_register_lzf()


# --- Public API --------------------------------------------------------------

from . import h5a, h5d, h5ds, h5f, h5fd, h5g, h5r, h5s, h5t, h5p, h5z

from ._hl import filters
from ._hl.base import is_hdf5, HLObject, Empty
from ._hl.files import (
    File,
    register_driver,
    unregister_driver,
    registered_drivers,
)
from ._hl.group import Group, SoftLink, ExternalLink, HardLink
from ._hl.dataset import Dataset
from ._hl.datatype import Datatype
from ._hl.attrs import AttributeManager

from .h5 import get_config
from .h5r import Reference, RegionReference
from .h5t import (special_dtype, check_dtype,
    vlen_dtype, string_dtype, enum_dtype, ref_dtype, regionref_dtype,
    check_vlen_dtype, check_string_dtype, check_enum_dtype, check_ref_dtype,
)

from ._ipython import (
    enable_ipython_completer, enable_ipython_formatter, enable_ipython)

from . import version
from .version import version as __version__


if version.hdf5_version_tuple[:3] >= get_config().vds_min_hdf5_version:
    from ._hl.vds import VirtualSource, VirtualLayout

if version.hdf5_version_tuple != version.hdf5_built_version_tuple:
    _warn(("h5py is running against HDF5 {0} when it was built against {1}, "
           "this may cause problems").format(
            '{0}.{1}.{2}'.format(*version.hdf5_version_tuple),
            '{0}.{1}.{2}'.format(*version.hdf5_built_version_tuple)
    ))


def run_tests(args=''):
    """Run tests with pytest and returns the exit status as an int.
    """
    # Lazy-loading of tests package to avoid strong dependency on test
    # requirements, e.g. pytest
    from .tests import run_tests
    return run_tests(args)


# --- Legacy API --------------------------------------------------------------

from .h5t import py_new_vlen as new_vlen
from .h5t import py_get_vlen as get_vlen
from .h5t import py_new_enum as new_enum
from .h5t import py_get_enum as get_enum

from .h5py_warnings import ModuleWrapper as _ModuleWrapper
highlevel = _ModuleWrapper("h5py.highlevel")
