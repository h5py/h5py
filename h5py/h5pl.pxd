# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2019 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

from .defs cimport *

cpdef void append(const char* search_path)
cpdef void prepend(const char* search_path)
cpdef void replace(const char* search_path, unsigned int index)
cpdef void insert(const char* search_path, unsigned int index)
cpdef void remove(unsigned int index)
cpdef object get(unsigned int index)
cpdef unsigned int size()
