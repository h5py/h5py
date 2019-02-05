# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2019 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    HDF5 plugin interface.
"""

include "config.pxi"
from utils cimport emalloc, efree

# === C API ===================================================================

IF HDF5_VERSION >= (1, 10, 1):
    cpdef append(const char* search_path):
        """(STRING search_path)"""
        H5PLappend(search_path)

    cpdef prepend(const char* search_path):
        """(STRING search_path)"""
        H5PLprepend(search_path)

    cpdef replace(const char* search_path, unsigned int index):
        """(STRING search_path, UINT index)"""
        H5PLreplace(search_path, index)

    cpdef insert(const char* search_path, unsigned int index):
        """(STRING search_path, UINT index)"""
        H5PLinsert(search_path, index)

    cpdef remove(unsigned int index):
        """(UINT index)"""
        H5PLremove(index)

    cpdef get(unsigned int index):
        """(UINT index) => STRING"""
        cpdef size_t n
        cpdef char* buf = NULL

        n = H5PLget(index, NULL, 0)
        buf = <char*>emalloc(sizeof(char)*(n + 1))
        try:
            H5PLget(index, buf, n + 1)
            return PyBytes_FromStringAndSize(buf, n)
        finally:
            efree(buf)

    cpdef size():
        """() => UINT"""
        cpdef unsigned int n = 0
        H5PLsize(&n)
        return n
