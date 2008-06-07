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

    def __str__(self):
        return '%2d:%2d "%s" at %s (%s: %s)' % (self.maj_num, self.min_num,
                self.desc, self.func_name, H5Eget_major(<H5E_major_t>self.maj_num),
                H5Eget_minor(<H5E_minor_t>self.min_num) )
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
    H5Ewalk(H5E_WALK_DOWNWARD, walk_cb, <void*>stack)
    return stack

def get_error_string():
    """ Return the HDF5 error stack as a single string.
    """
    cdef int stacklen
    cdef H5ErrorStackElement el

    stack = get_error_stack()
    stacklen = len(stack)

    if stacklen == 0:
        msg = "Unspecified HDF5 error"
    else:
        el = stack[0]
        msg = "%s (%s: %s)" % (el.desc.capitalize(), el.func_name, 
                                H5Eget_major(<H5E_major_t>el.maj_num))
        if stacklen > 1:
            msg = msg + "\nHDF5 Error Stack:"
            for i from 0<=i<stacklen:
                msg = msg + '\n' + str(stack[i])
                #el = stack[i]
                #msg = msg + '\n    %d: "%s" at %s' % \
                #            (i, el.desc.capitalize(), el.func_name)

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

    # Highest priority: really bad errors
    if mn == H5E_UNSUPPORTED:
        exc = NotImplementedError
    elif mj == H5E_IO:
        exc = IOError

    # Map function argument exceptions to native Python exceptions.
    # H5E_BADTYPE does not raise TypeError as this is too easily confused
    # with the results of Pyrex auto-validation.
    elif mj == H5E_ARGS and (mn == H5E_BADRANGE or mn == H5E_BADVALUE \
                             or mn == H5E_BADTYPE):
        exc = ValueError

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

    return 1

cdef int _enable_exceptions() except -1:
    # Enable automatic exception handling, by registering the above callback
    if H5Eset_auto(err_callback, NULL) < 0:
        raise RuntimeError("Failed to register HDF5 exception callback.")
    return 0

cdef int _disable_exceptions() except -1:
    # Disable automatic exception handling
    if H5Eset_auto(NULL, NULL) < 0:
        raise RuntimeError("Failed to unregister HDF5 exception callback.")
    return 0

cdef err_c pause_errors() except NULL:
    # Temporarily disable automatic exception handling, and return a cookie
    # which can later be used to re-enable it.
    cdef err_c cookie
    cdef void* whatever
    cookie = NULL

    if H5Eget_auto(&cookie, &whatever) < 0:
        raise RuntimeError("Failed to retrieve the current error handler.")

    if H5Eset_auto(NULL, NULL) < 0:
        raise RuntimeError("Failed to temporarily disable error handling.")

    return cookie

cdef int resume_errors(err_c cookie) except -1:
    # Resume automatic exception handling, using a cookie from a previous
    # call to pause_errors().  Also clears the error stack.
    if H5Eset_auto(cookie, NULL) < 0:
        raise RuntimeError("Failed to re-enable error handling.")
    
    if H5Eclear() < 0:
        raise RuntimeError("Failed to clear error stack.")

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











