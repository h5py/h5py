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

from python cimport PyExc_Exception, PyErr_SetString

# === Public exception hierarchy ==============================================

class H5Error(StandardError):
    """ Base class for internal HDF5 library exceptions
    """
    pass

class ConversionError(StandardError):
    """ Indicates error on Python side of dtype transformation.
    """
    pass

class FileError(H5Error):
    """ HDF5 file I/0 error
    """
    pass

class GroupError(H5Error):
    pass

class DataspaceError(H5Error):
    pass

class DatatypeError(H5Error):
    pass

class DatasetError(H5Error):
    pass

class PropertyError(H5Error):
    pass

class H5AttributeError(H5Error):
    pass

class FilterError(H5Error):
    pass

class IdentifierError(H5Error):
    pass

class H5ReferenceError(H5Error):
    pass


# === Error stack inspection ==================================================

cdef class H5ErrorStackElement:
    """
        Represents an entry in the HDF5 error stack.
        Loosely modeled on the H5E_error_t struct.
    """
    cdef readonly int maj_num
    cdef readonly int min_num
    cdef readonly object func_name
    cdef readonly object file_name
    cdef readonly unsigned int line
    cdef readonly object desc

cdef herr_t walk_cb(int n, H5E_error_t *err_desc, void* stack_in):
    # Callback function to extract elements from the HDF5 error stack

    stack = <object>stack_in
    cdef H5ErrorStackElement element

    element = H5ErrorStackElement()
    element.maj_num = err_desc.maj_num
    element.min_num = err_desc.min_num
    element.func_name = err_desc.func_name
    element.file_name = err_desc.file_name
    element.desc = err_desc.desc

    stack.append(element)

    return 0

def get_error_stack():

    stack = []
    H5Ewalk(H5E_WALK_UPWARD, walk_cb, <void*>stack)
    return stack

def get_error_string():
    """ Return the HDF5 error stack as a single string.
    """
    cdef int stacklen
    stack = get_error_stack()
    stacklen = len(stack)
    if stacklen == 0:
        msg = "Unspecified HDF5 error"
    else:
        msg = "%s (%s)" % (stack[0].desc.capitalize(), stack[0].func_name)
        if stacklen > 1:
            msg = msg + "\nHDF5 Error Stack:\n"
            for i from 0<=i<stacklen:
                el = stack[i]
                msg = msg + '    %d: "%s" at %s\n' % (i, el.desc.capitalize(), el.func_name)

    return msg


# === Automatic exception API =================================================

cdef herr_t extract_cb(int n, H5E_error_t *err_desc, void* data_in):
    # Callback to determine error information at top/bottom of stack
    cdef H5E_error_t *err_struct
    err_struct = <H5E_error_t*>data_in
    err_struct.maj_num = err_desc.maj_num
    err_struct.min_num = err_desc.min_num
    return 1
    
cdef herr_t err_callback(void* client_data):
    # Callback which sets Python exception based on the current error stack.

    # Can't use the standard Pyrex raise because then the traceback
    # points here!

    cdef H5E_error_t err_struct
    cdef H5E_major_t mj
    cdef H5E_minor_t mn

    # Determine the error numbers for the first entry on the stack.
    H5Ewalk(H5E_WALK_UPWARD, extract_cb, &err_struct)
    mj = err_struct.maj_num
    mn = err_struct.min_num

    # Most common minor errors
    if mn == H5E_UNSUPPORTED:
        exc = NotImplementedError
    elif mn == H5E_BADTYPE:
        exc = TypeError
    elif mn == H5E_BADRANGE or mn == H5E_BADVALUE:
        exc = ValueError

    # Major errors which map to native Python exceptions
    elif mj == H5E_IO:
        exc = IOError

    # Major errors which map to new h5e exception classes
    elif mj == H5E_FILE:
        exc = FileError
    elif mj == H5E_DATATYPE:
        exc = DatatypeError
    elif mj == H5E_DATASPACE:
        exc = DataspaceError
    elif mj == H5E_DATASET:
        exc = DatasetError
    elif mj == H5E_PLIST:
        exc = PropertyListError
    elif mj == H5E_ATTR:
        exc = H5AttributeError
    elif mj == H5E_PLINE:
        exc = FilterError
    elif mj == H5E_REFERENCE:
        exc = H5ReferenceError

    # Catchall: base H5Error
    else:
        exc = H5Error

    msg = get_error_string()
    PyErr_SetString(exc, msg)


def _enable_exceptions():
    if H5Eset_auto(err_callback, NULL) < 0:
        raise RuntimeError("Failed to register HDF5 exception callback.")

def _disable_exceptions():
    if H5Eset_auto(NULL, NULL) < 0:
        raise RuntimeError("Failed to unregister HDF5 exception callback.")

cdef err_c pause_errors() except NULL:
    cdef err_c cookie
    cdef void* whatever
    cdef herr_t retval
    cookie = NULL

    retval = H5Eget_auto(&cookie, &whatever)
    if retval < 0:
        raise RuntimeError("Failed to retrieve the current error handler.")

    retval = H5Eset_auto(NULL, NULL)
    if retval < 0:
        raise RuntimeError("Failed to temporarily disable error handling.")

    return cookie

cdef int resume_errors(err_c cookie) except -1:
    cdef herr_t retval
    retval = H5Eset_auto(cookie, NULL)
    if retval < 0:
        raise RuntimeError()
    return 0

# --- temporary test functions ---

cdef extern from "hdf5.h":
  htri_t H5Pexist( hid_t id,  char *name) except *
  herr_t H5Tclose(hid_t type_id  ) except *
  hid_t H5T_STD_I8LE

def test_error():
    H5Pexist(-1, "foobar")

def test_2():
    H5Tclose(H5T_STD_I8LE)

cdef void ctest3() except *:
    PyErr_SetString(Foobar, "foo")

def test_3():
    ctest3()











