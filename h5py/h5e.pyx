from h5 cimport herr_t, htri_t


cdef extern from "Python.h":

  ctypedef extern class __builtin__.BaseException [object PyBaseExceptionObject]:
    cdef object dict
    cdef object args
    cdef object message


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


cdef class ErrorStackElement:
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

    
cdef herr_t walk_cb(int n, H5E_error_t *err_desc, stack):
    # Callback function to extract elements from the HDF5 error stack

    cdef ErrorStackElement element

    element = ErrorStackElement()
    element.maj_num = err_desc.maj_num
    element.min_num = err_desc.min_num
    element.func_name = err_desc.func_name
    element.file_name = err_desc.file_name
    element.desc = err_desc.desc

    stack.append(element)

    return 0

cdef class H5LibraryError(H5Error):
    """
        Base class for exceptions which include an HDF5 library traceback.
        Upon instantiation, takes a snapshot of the HDF5 error stack and 
        stores it internally.
    """
    cdef readonly object hdf5_stack
    
    def __init__(self, *args):
        cdef int i

        H5Error.__init__(self)
        stack = []
        H5Ewalk(H5E_WALK_DOWNWARD, walk_cb, stack)
        self.hdf5_stack = stack

        # Stringify the stack
        if len(stack) == 0:
            msg = "Unspecified HDF5 error"
        else:
            msg = stack[0].desc.capitalize() + "\nHDF5 Error Stack:\n"
            for i from 0<=i<len(stack):
                el = stack[i]
                msg = msg + '    %d: "%s" at %s' % (i, el.desc.capitalize(), el.func_name)

        self.args = (msg,)

cdef extern from "Python.h":
  void PyErr_SetString(object type_, object msg)
  void PyErr_SetNone(object type_)

cdef herr_t err_callback(void* client_data):
    # Callback which does nothing but set a Python exception
    # Can't use the standard Pyrex raise because then the traceback
    # points here!
    PyErr_SetNone(H5LibraryError)

def enable_exceptions():
    if H5Eset_auto(err_callback, NULL) < 0:
        raise RuntimeError("Failed to register HDF5 exception callback.")

def disable_exceptions():
    if H5Eset_auto(NULL, NULL) < 0:
        raise RuntimeError("Failed to unregister HDF5 exception callback.")

cdef extern from "hdf5.h":
  htri_t H5Pexist( hid_t id,  char *name) except? -1

def test_error():
    H5Pexist(-1, "foobar")













