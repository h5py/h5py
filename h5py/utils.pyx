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

from python cimport PyTuple_Check, PyList_Check, PyErr_SetString

cdef int require_tuple(object tpl, int none_allowed, int size, char* name) except -1:
    # Ensure that tpl is in fact a tuple, or None if none_allowed is nonzero.
    # If size >= 0, also ensure that the length matches.
    # Otherwise raises ValueError

    if (tpl is None and none_allowed) or \
      ( PyTuple_Check(tpl) and (size < 0 or len(tpl) == size)):
        return 1

    nmsg = ""
    smsg = ""
    if size >= 0:
        smsg = " of size %d" % size
    if none_allowed:
        nmsg = " or None"

    msg = "%s must be a tuple%s%s." % (name, smsg, nmsg)
    PyErr_SetString(ValueError, msg)
    return -1

cdef int require_list(object lst, int none_allowed, int size, char* name) except -1:
    # Counterpart of require_tuple, for lists

    if (lst is None and none_allowed) or \
      ( PyList_Check(lst) and (size < 0 or len(lst) == size)):
        return 1

    nmsg = ""
    smsg = ""
    if size >= 0:
        smsg = " of size %d" % size
    if none_allowed:
        nmsg = " or None"

    msg = "%s must be a list%s%s." % (name, smsg, nmsg)
    PyErr_SetString(ValueError, msg)
    return -1





