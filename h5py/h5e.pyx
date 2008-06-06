from h5 cimport herr_t, htri_t


cdef extern from "Python.h":

  ctypedef extern class __builtin__.BaseException [object PyBaseExceptionObject]:
    cdef object dict
    cdef object args
    cdef object message

cdef extern from "Python.h":
  void PyErr_SetString(object type_, object msg)
  void PyErr_SetNone(object type_)

# === Base exception hierarchy ================================================

cdef class H5Error(BaseException):
    """
        Base class for all HDF5 exceptions.
    """
    pass


cdef class ConversionError(H5Error):
    """
        Represents a Python-side error performing data conversion between
        Numpy arrays or dtypes and their HDF5 equivalents.
    """
    pass


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

cdef class LibraryError(H5Error):
    """
        Base class for exceptions which include an HDF5 library traceback.
        Upon instantiation, takes a snapshot of the HDF5 error stack and 
        stores it internally.
    """
    cdef readonly object hdf5_stack
    
    def __init__(self, *args):
        cdef int i
        cdef int stacklen

        H5Error.__init__(self)
        stack = []
        H5Ewalk(H5E_WALK_DOWNWARD, walk_cb, <void*>stack)
        self.hdf5_stack = stack

        # Stringify the stack
        stacklen = len(stack)
        if stacklen == 0:
            msg = "Unspecified HDF5 error"
        else:
            msg = "%s (%s)" % (stack[0].desc.capitalize(), stack[0].func_name)
            if stacklen > 1:
                msg = msg + "\nHDF5 Error Stack:\n"
                for i from 0<=i<stacklen:
                    el = stack[i]
                    msg = msg + '    %d: "%s" at %s' % (i, el.desc.capitalize(), el.func_name)

        self.args = (msg,)

# === Public exception classes ================================================

cdef class InternalError(LibraryError):
    """
        Catchall class for major error numbers which don't have their
        own exception class.
    """
    pass

cdef class InvalidArgsError(LibraryError):
    pass

cdef class DatatypeError(LibraryError):
    pass

cdef class DataspaceError(LibraryError):
    pass

cdef class DatasetError(LibraryError):
    pass

cdef class StorageError(LibraryError):
    pass

cdef class PropertyListError(LibraryError):
    pass

cdef class AttributeError_H5(LibraryError):
    pass

cdef class FilterError_H5(LibraryError):
    pass


# === Automatic exception API =================================================

cdef herr_t maj_cb(int n, H5E_error_t *err_desc, void* num_in):
    # Callback to determine the major error number at either the top
    # or bottom of the stack
    cdef H5E_major_t *num
    num = <H5E_major_t*>num_in
    num[0] = err_desc.maj_num
    return 1
    
cdef herr_t err_callback(void* client_data):
    # Callback which does nothing but set a Python exception
    # Can't use the standard Pyrex raise because then the traceback
    # points here!

    cdef H5E_major_t num

    # Determine the major error number for the first entry on the stack.
    H5Ewalk(H5E_WALK_DOWNWARD, maj_cb, &num)

    exc = InternalError
    if num == H5E_ARGS:
        exc = InvalidArgsError
    elif num == H5E_DATATYPE:
        exc = DatatypeError
    elif num == H5E_DATASPACE:
        exc = DataspaceError
    elif num == H5E_DATASET:
        exc = DatasetError
    elif num == H5E_STORAGE:
        exc = StorageError
    elif num == H5E_PLIST:
        exc = PropertyListError
    elif num == H5E_ATTR:
        exc = AttributeError_H5
    elif num == H5E_PLINE:
        exc = FilterError_H5

    PyErr_SetNone(exc)

def enable_exceptions():
    if H5Eset_auto(err_callback, NULL) < 0:
        raise RuntimeError("Failed to register HDF5 exception callback.")

def disable_exceptions():
    if H5Eset_auto(NULL, NULL) < 0:
        raise RuntimeError("Failed to unregister HDF5 exception callback.")

# --- temporary test functions ---

cdef extern from "hdf5.h":
  htri_t H5Pexist( hid_t id,  char *name) except *
  herr_t H5Tclose(hid_t type_id  ) except *
  hid_t H5T_STD_I8LE

def test_error():
    H5Pexist(-1, "foobar")

def test_2():
    H5Tclose(H5T_STD_I8LE)












