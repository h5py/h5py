# cython: language_level=3
# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

from .defs cimport *

cdef herr_t attr_rw(hid_t attr, hid_t mtype, void *progbuf, int read) except -1

cdef herr_t dset_rw(size_t count, hid_t* _dset, hid_t* _mtype, hid_t* _mspace, hid_t* _fspace,
                    hid_t dxpl, void **progbuf, int read) except -1

cdef htri_t needs_bkg_buffer(hid_t src, hid_t dst) except -1
