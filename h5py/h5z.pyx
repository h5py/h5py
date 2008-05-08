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


"""
    Filter API and constants
"""

# Pyrex compile-time imports
from h5  cimport herr_t, htri_t

# Runtime imports
import h5
from h5 import DDict
from errors import FilterError

# === Public constants and data structures ====================================

FILTER_ERROR    = H5Z_FILTER_ERROR
FILTER_NONE     = H5Z_FILTER_NONE
FILTER_ALL      = H5Z_FILTER_ALL
FILTER_DEFLATE  = H5Z_FILTER_DEFLATE
FILTER_SHUFFLE  = H5Z_FILTER_SHUFFLE
FILTER_FLETCHER32 = H5Z_FILTER_FLETCHER32
FILTER_SZIP     = H5Z_FILTER_SZIP
FILTER_RESERVED = H5Z_FILTER_RESERVED
FILTER_MAX      = H5Z_FILTER_MAX
FILTER_NMAX     = H5Z_MAX_NFILTERS
_FILTER_MAPPER = { H5Z_FILTER_ERROR: 'ERROR', H5Z_FILTER_NONE: 'NONE',
                   H5Z_FILTER_ALL: 'ALL', H5Z_FILTER_DEFLATE: 'DEFLATE',
                   H5Z_FILTER_SHUFFLE: 'SHUFFLE', H5Z_FILTER_FLETCHER32: 'FLETCHER32',
                   H5Z_FILTER_SZIP: 'SZIP', H5Z_FILTER_RESERVED: 'RESERVED'}
FILTER_MAPPER = DDict(_FILTER_MAPPER)

FLAG_DEFMASK    = H5Z_FLAG_DEFMASK
FLAG_MANDATORY  = H5Z_FLAG_MANDATORY
FLAG_OPTIONAL   = H5Z_FLAG_OPTIONAL
FLAG_INVMASK    = H5Z_FLAG_INVMASK
FLAG_REVERSE    = H5Z_FLAG_REVERSE
FLAG_SKIP_EDC   = H5Z_FLAG_SKIP_EDC
_FLAG_MAPPER = {H5Z_FLAG_DEFMASK: 'DEFMASK', H5Z_FLAG_MANDATORY: 'MANDATORY',
                H5Z_FLAG_OPTIONAL: 'OPTIONAL', H5Z_FLAG_INVMASK: 'INVMASK',
                H5Z_FLAG_REVERSE: 'REVERSE', H5Z_FLAG_SKIP_EDC: 'SKIP EDC' }
FLAG_MAPPER = DDict(_FLAG_MAPPER)

#skip SZIP options

CONFIG_ENCODE_ENABLED = H5Z_FILTER_CONFIG_ENCODE_ENABLED
CONFIG_DECODE_ENABLED = H5Z_FILTER_CONFIG_DECODE_ENABLED
_CONFIG_MAPPER = { H5Z_FILTER_CONFIG_ENCODE_ENABLED: 'ENCODE ENABLED',
                   H5Z_FILTER_CONFIG_DECODE_ENABLED: 'ENCODE DISABLED' }
CONFIG_MAPPER = DDict(_CONFIG_MAPPER)

EDC_ERROR   = H5Z_ERROR_EDC
EDC_DISABLE = H5Z_DISABLE_EDC
EDC_ENABLE  = H5Z_ENABLE_EDC
EDC_NONE    = H5Z_NO_EDC
_EDC_MAPPER = { H5Z_ERROR_EDC: 'ERROR', H5Z_DISABLE_EDC: 'DISABLE EDC',
                H5Z_ENABLE_EDC: 'ENABLE EDC', H5Z_NO_EDC: 'NO EDC' }
EDC_MAPPER = DDict(_EDC_MAPPER)

# === Filter API  =============================================================

def filter_avail(int filter_id):

    cdef htri_t retval
    retval = H5Zfilter_avail(<H5Z_filter_t>filter_id)
    if retval < 0:
        raise FilterError("Can't determine availability of filter %d" % filter_id)
    return bool(retval)

def get_filter_info(int filter_id):

    cdef herr_t retval
    cdef unsigned int flags
    retval = H5Zget_filter_info(<H5Z_filter_t>filter_id, &flags)
    if retval < 0:
        raise FilterError("Can't determine flags of filter %d" % filter_id)
    return flags














