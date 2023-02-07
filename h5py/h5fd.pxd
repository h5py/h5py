# cython: language_level=3
# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

# This file contains code or comments from the HDF5 library.  See the file
# licenses/hdf5.txt for the full HDF5 software license.

from .defs cimport *

cdef class IOCConfig:
    cdef H5FD_ioc_config_t ioc_config

cdef class SubfilingConfig:
    cdef H5FD_subfiling_config_t subf_config
