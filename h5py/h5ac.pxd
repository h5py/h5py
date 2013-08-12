#+
#
# This file is part of h5py, a low-level Python interface to the HDF5 library.
#
# Copyright (C) 2013 Andrew Collette
# License: BSD  (See LICENSE.txt for full license)
#
# $Date$
#
#-

from defs cimport *

cdef class CacheConfig:
    cdef H5AC_cache_config_t cache_config
