# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Filter API and constants.
"""

from ._objects import phil, with_phil


# === Public constants and data structures ====================================

FILTER_LZF = H5PY_FILTER_LZF

FILTER_ERROR    = H5Z_FILTER_ERROR
FILTER_NONE     = H5Z_FILTER_NONE
FILTER_ALL      = H5Z_FILTER_ALL
FILTER_DEFLATE  = H5Z_FILTER_DEFLATE
FILTER_SHUFFLE  = H5Z_FILTER_SHUFFLE
FILTER_FLETCHER32 = H5Z_FILTER_FLETCHER32
FILTER_SZIP     = H5Z_FILTER_SZIP
FILTER_NBIT     = H5Z_FILTER_NBIT
FILTER_SCALEOFFSET = H5Z_FILTER_SCALEOFFSET
FILTER_RESERVED = H5Z_FILTER_RESERVED
FILTER_MAX      = H5Z_FILTER_MAX

FLAG_DEFMASK    = H5Z_FLAG_DEFMASK
FLAG_MANDATORY  = H5Z_FLAG_MANDATORY
FLAG_OPTIONAL   = H5Z_FLAG_OPTIONAL
FLAG_INVMASK    = H5Z_FLAG_INVMASK
FLAG_REVERSE    = H5Z_FLAG_REVERSE
FLAG_SKIP_EDC   = H5Z_FLAG_SKIP_EDC

SZIP_ALLOW_K13_OPTION_MASK  = H5_SZIP_ALLOW_K13_OPTION_MASK   #1
SZIP_CHIP_OPTION_MASK       = H5_SZIP_CHIP_OPTION_MASK        #2
SZIP_EC_OPTION_MASK         = H5_SZIP_EC_OPTION_MASK          #4
SZIP_NN_OPTION_MASK         = H5_SZIP_NN_OPTION_MASK          #32
SZIP_MAX_PIXELS_PER_BLOCK   = H5_SZIP_MAX_PIXELS_PER_BLOCK    #32

SO_FLOAT_DSCALE = H5Z_SO_FLOAT_DSCALE
SO_FLOAT_ESCALE = H5Z_SO_FLOAT_ESCALE
SO_INT          = H5Z_SO_INT
SO_INT_MINBITS_DEFAULT = H5Z_SO_INT_MINBITS_DEFAULT

FILTER_CONFIG_ENCODE_ENABLED = H5Z_FILTER_CONFIG_ENCODE_ENABLED
FILTER_CONFIG_DECODE_ENABLED = H5Z_FILTER_CONFIG_DECODE_ENABLED

ERROR_EDC   = H5Z_ERROR_EDC
DISABLE_EDC = H5Z_DISABLE_EDC
ENABLE_EDC  = H5Z_ENABLE_EDC
NO_EDC      = H5Z_NO_EDC


# === Filter API  =============================================================

@with_phil
def filter_avail(int filter_code):
    """(INT filter_code) => BOOL

    Determine if the given filter is available to the library. The
    filter code should be one of:

    - FILTER_DEFLATE
    - FILTER_SHUFFLE
    - FILTER_FLETCHER32
    - FILTER_SZIP
    """
    return <bint>H5Zfilter_avail(<H5Z_filter_t>filter_code)


@with_phil
def get_filter_info(int filter_code):
    """(INT filter_code) => INT filter_flags

    Retrieve a bitfield with information about the given filter. The
    filter code should be one of:

    - FILTER_DEFLATE
    - FILTER_SHUFFLE
    - FILTER_FLETCHER32
    - FILTER_SZIP

    Valid bitmasks for use with the returned bitfield are:

    - FILTER_CONFIG_ENCODE_ENABLED
    - FILTER_CONFIG_DECODE_ENABLED
    """
    cdef unsigned int flags
    H5Zget_filter_info(<H5Z_filter_t>filter_code, &flags)
    return flags

def _register_lzf():
    register_lzf()









