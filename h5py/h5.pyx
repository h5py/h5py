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
    Common support and versioning module for the h5py HDF5 interface.

    This is an internal module which is designed to set up the library and
    enable HDF5 exception handline.  It also enables debug logging, if the
    library has been compiled with a nonzero debugging level.

    Useful things defined here:
    
      version               String with h5py version (e.g. "0.2.0")
      version_tuple         3-tuple with h5py version (e.g. (0, 2, 0))

      hdf5_version:         String with library version (e.g. "1.6.5")
      hdf5_version_tuple:   3-tuple form of the version (e.g. (1,6,5))

      api_version:          String form of the API used (e.g. "1.6")
      api_version_tuple:    2-tuple form of the version (e.g. (1,6))

    All exception classes and error handling functions are also in this module.
"""

include "conditions.pxi"
from python cimport PyErr_SetObject

import atexit
import threading
from weakref import WeakKeyDictionary

# Logging is only enabled when compiled with H5PY_DEBUG nonzero
IF H5PY_DEBUG:
    import logging
    logger = logging.getLogger('h5py')
    logger.setLevel(H5PY_DEBUG)
    logger.addHandler(logging.StreamHandler())

def get_libversion():
    """ () => TUPLE (major, minor, release)

        Retrieve the HDF5 library version as a 3-tuple.
    """
    cdef unsigned int major
    cdef unsigned int minor
    cdef unsigned int release
    cdef herr_t retval
    
    H5get_libversion(&major, &minor, &release)

    return (major, minor, release)

def _close():
    """ Internal function; do not call unless you want to lose all your data.
    """
    H5close()

def _open():
    """ Internal function; do not call unless you want to lose all your data.
    """
    H5open()

cdef class H5PYConfig:

    """
        Global configuration object for the h5py package.

        Properties:
        lock            Global reentrant lock for threading support
        compile_opts    Dictionary of compile-time flags
    """

    def __init__(self):
        self._complex_names = ('r','i')
        self.compile_opts = {'IO_NONBLOCK': H5PY_NONBLOCK,
                             'DEBUG': H5PY_DEBUG}
        self._lock = threading.RLock()

    property lock:
        """ Reentrant lock instance; default is threading.RLock().
            
            Whatever you provide MUST support the Python context manager
            protocol, and provide the methods acquire() and release().  It
            also MUST be reentrant, or dataset reads/writes will deadlock.
        """
        def __get__(self):
            return self._lock

        def __set__(self, val):
            if not (hasattr(val, 'acquire') and hasattr(val, 'release') and\
                    hasattr(val, '__enter__') and hasattr(val, '__exit__')):
                raise ValueError("Generated locks must provide __enter__, __exit__, acquire, release")
            current_lock = self._lock
            current_lock.acquire()
            try:
                self._lock = val
            finally:
                current_lock.release()

    property complex_names:
        """ Unused
        """
        def __get__(self):
            return self._complex_names

        def __set__(self, val):
            # TODO: validation
            self._complex_names = val



cdef object standard_richcmp(object self, object other, int how):
    # This needs to be shared because of weird CPython quirks involving
    # subclasses and the __hash__ method.

    if how == 2 or how == 3:
        
        if not typecheck(self, ObjectID) and typecheck(other, ObjectID):
            return NotImplemented

        eq = (hash(self) == hash(other))

        if how == 2:
            return eq
        return not eq

    return NotImplemented

cdef class ObjectID:

    """
        Base class for all HDF5 identifiers.

        This is an extremely thin object layer, which makes dealing with
        HDF5 identifiers a less frustrating experience.  It synchronizes
        Python object reference counts with their HDF5 counterparts, so that
        HDF5 identifiers are automatically closed when they become unreachable.

        The only (known) HDF5 property which can problematic is locked objects;
        there is no way to determine whether or not an HDF5 object is locked
        or not, without trying an operation and having it fail.  A "lock" flag
        is maintained on the Python side, and is set by methods like
        TypeID.lock(), but this is not tracked across copies.  Until HDF5
        provides something like H5Tis_locked(), this will not be fixed.

        The truth value of an ObjectID (i.e. bool(obj_id)) indicates whether
        the underlying HDF5 identifier is valid.
    """

    property _valid:
        """ Indicates whether or not this identifier points to an HDF5 object.
        """
        def __get__(self):
            return H5Iget_type(self.id) != H5I_BADID

    def __nonzero__(self):
        """ Truth value for object identifiers (like _valid) """
        return self._valid

    def __cinit__(self, hid_t id_):
        """ Object init; simply records the given ID. """
        self._locked = 0
        self.id = id_
        IF H5PY_DEBUG:
            logger.debug("+ %s" % str(self))

    def __dealloc__(self):
        """ Automatically decrefs the ID, if it's valid. """
        IF H5PY_DEBUG:
            logger.debug("- %s" % str(self))
        if (not self._locked) and self._valid:
            H5Idec_ref(self.id)

    def __copy__(self):
        """ Create another object wrapper which points to the same id. 

            WARNING: Locks (i.e. datatype lock() methods) do NOT work correctly
            across copies.
        """
        cdef ObjectID copy
        copy = type(self)(self.id)
        assert typecheck(copy, ObjectID), "ObjectID copy encountered invalid type"
        if self._valid and not self._locked:
            H5Iinc_ref(self.id)
        copy._locked = self._locked
        IF H5PY_DEBUG:
            logger.debug("c %s" % str(self))
        return copy

    def __richcmp__(self, object other, int how):
        """ Supports only == and != """
        return standard_richcmp(self, other, how)

    def __hash__(self):
        """ Hash method defaults to the identifer, as this cannot change over
            the life of the object.
        """
        return self.id

    def __str__(self):
        if self._valid:
            ref = str(H5Iget_ref(self.id))
        else:
            ref = "X"

        if self._locked:
            lstr = "L"
        else:
            lstr = "U"

        return "%9d [%s] (%s) %s" % (self.id, ref, lstr, self.__class__.__name__)

    def __repr__(self):
        return self.__str__()

# === Public exception hierarchy ==============================================

class H5Error(Exception):
    """ Base class for internal HDF5 library exceptions.
        Subclass of EnvironmentError; errno is computed from the HDF5 major
        and minor error numbers:
            1000*(major number) + minor number
    """
    pass

# --- New classes -------------------------------------------------------------

           
#    H5E_ARGS,                   # invalid arguments to routine
class ArgsError(H5Error):
    """ H5E_ARGS """
    pass
        
#    H5E_RESOURCE,               # resource unavailable   
class ResourceError(H5Error):
    """ H5E_RESOURCE """
    pass
                    
#   H5E_INTERNAL,               #  Internal error (too specific to document)
class InternalError(H5Error):
    """ H5E_INTERNAL """
    pass

#    H5E_FILE,                   # file Accessability       
class FileError(H5Error):
    """ H5E_FILE """
    pass
                  
#    H5E_IO,                     # Low-level I/O                      
class LowLevelIOError(H5Error):
    """ H5E_IO """
    pass
        
#    H5E_FUNC,                   # function Entry/Exit     
class FuncError(H5Error):
    """ H5E_FUNC """
    pass
                   
#    H5E_ATOM,                   # object Atom         
class AtomError(H5Error):
    """ H5E_ATOM """
    pass
                       
#   H5E_CACHE,                  # object Cache        
class CacheError(H5Error):
    """ H5E_CACHE """
    pass
                       
#   H5E_BTREE,                  # B-Tree Node           
class BtreeError(H5Error):
    """ H5E_BTREE """
    pass
                     
#    H5E_SYM,                    # symbol Table         
class SymbolError(H5Error):
    """ H5E_SYM """
    pass
                      
#   H5E_HEAP,                   # Heap  
class HeapError(H5Error):
    """ H5E_HEAP """
    pass
                                     
#    H5E_OHDR,                   # object Header     
class ObjectHeaderError(H5Error):
    """ H5E_OHDR """
    pass
                         
#    H5E_DATATYPE,               # Datatype    
class DatatypeError(H5Error):
    """ H5E_DATATYPE """
    pass
                               
#    H5E_DATASPACE,              # Dataspace                      
class DataspaceError(H5Error):
    """ H5E_DATASPACE """
    pass
            
#    H5E_DATASET,                # Dataset                        
class DatasetError(H5Error):
    """ H5E_DATASET """
    pass
            
#    H5E_STORAGE,                # data storage                
class StorageError(H5Error):
    """ H5E_STORAGE """
    pass
               
#    H5E_PLIST,                  # Property lists                  
class PropertyError(H5Error):
    """ H5E_PLIST """
    pass
           
#    H5E_ATTR,                   # Attribute     
class AttrError(H5Error):
    """ H5E_ATTR """
    pass
                             
#    H5E_PLINE,                  # Data filters       
class FilterError(H5Error):
    """ H5E_PLINE """
    pass
                        
#    H5E_EFL,                    # External file list                         
class FileListError(H5Error):
    """ H5E_EFL """
    pass

#    H5E_REFERENCE,              # References          
class RefError(H5Error):
    """ H5E_REFERENCE """
    pass
                       
#    H5E_VFL,                    # Virtual File Layer        
class VirtualFileError(H5Error):
    """ H5E_VFL """
    pass
         
# H5E_TBBT removed; does not appear in 1.8.X

#    H5E_TST,                    # Ternary Search Trees                       
class TSTError(H5Error):
    """ H5E_TST """
    pass

#    H5E_RS,                     # Reference Counted Strings        
class RSError(H5Error):
    """ H5E_RS """
    pass
          
#    H5E_ERROR,                  # Error API                    
class ErrorError(H5Error):
    """ H5E_ERROR """
    pass
              
#    H5E_SLIST                   # Skip Lists        
class SkipListError(H5Error):
    """ H5E_SLIST """
    pass  

_exceptions = {
    H5E_ARGS: ArgsError,
    H5E_RESOURCE: ResourceError,
    H5E_INTERNAL: InternalError,
    H5E_FILE: FileError,
    H5E_IO: LowLevelIOError,
    H5E_FUNC: FuncError,
    H5E_ATOM: AtomError,
    H5E_CACHE: CacheError,
    H5E_BTREE: BtreeError,
    H5E_SYM: SymbolError,
    H5E_HEAP: HeapError,
    H5E_OHDR: ObjectHeaderError,
    H5E_DATATYPE: DatatypeError,
    H5E_DATASPACE: DataspaceError,
    H5E_DATASET: DatasetError,
    H5E_STORAGE: StorageError,
    H5E_PLIST: PropertyError,
    H5E_ATTR: AttrError,
    H5E_PLINE: FilterError,
    H5E_EFL: FileListError,
    H5E_REFERENCE: RefError,
    H5E_VFL: VirtualFileError,
    H5E_TST: TSTError,
    H5E_RS: RSError,
    H5E_ERROR: ErrorError,
    H5E_SLIST: SkipListError}

# === Error stack inspection ==================================================

cdef class ErrorStackElement:
    """
        Represents an entry in the HDF5 error stack.
        Modeled on the H5E_error_t struct.  All properties are read-only.

        Atributes
        maj_num:    INT major error number
        min_num:    INT minor error number
        func_name:  STRING name of failing function
        file_name:  STRING name of file in which error occurreed
        line:       UINT line number at which error occured
        desc:       STRING description of error
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
    cdef ErrorStackElement element

    element = ErrorStackElement()
    element.maj_num = err_desc.maj_num
    element.min_num = err_desc.min_num
    element.func_name = err_desc.func_name
    element.file_name = err_desc.file_name
    element.desc = err_desc.desc

    stack.append(element)

    return 0

def error_stack():
    """ () => LIST error_stack

        Retrieve the HDF5 error stack as a list of ErrorStackElement objects,
        with the most recent call (the deepest one) listed last.
    """
    stack = []
    H5Ewalk(H5E_WALK_DOWNWARD, walk_cb, <void*>stack)
    return stack

def error_string():
    """ () => STRING error_stack

        Return a string representation of the current error condition.
        Format is one line of the format 
            '<Description> (<Function name>: <error type>)'

        If the stack is more than one level deep, this is followed by a
        header and then n lines of the format:
            '    n: "<Description>" at <function name>'
    """
    cdef int stacklen
    cdef ErrorStackElement el

    stack = error_stack()
    stacklen = len(stack)

    if stacklen == 0:
        msg = "Unspecified HDF5 error"
    else:
        el = stack[0]
        msg = "%s (%s)" % (el.desc.capitalize(), el.func_name)
        if stacklen > 1:
            msg = msg + "\nHDF5 Error Stack:"
            for i from 0<=i<stacklen:
                #msg = msg + '\n' + str(stack[i])
                el = stack[i]
                msg = msg + '\n    %d: "%s" at %s' % \
                            (i, el.desc.capitalize(), el.func_name)

    return msg

def clear():
    """ ()

        Clear the error stack.
    """
    H5Eclear()

def get_major(int error):
    """ (INT error) => STRING description

        Get a description associated with an HDF5 minor error code.
    """
    return H5Eget_major(<H5E_major_t>error)

def get_minor(int error):
    """ (INT error) => STRING description

        Get a description associated with an HDF5 minor error code.
    """
    return H5Eget_minor(<H5E_minor_t>error)

# === Automatic exception API =================================================

cdef herr_t extract_cb(int n, H5E_error_t *err_desc, void* data_in):
    # Callback to determine error information at top/bottom of stack
    cdef H5E_error_t *err_struct
    err_struct = <H5E_error_t*>data_in
    err_struct.maj_num = err_desc.maj_num
    err_struct.min_num = err_desc.min_num
    return 1
    
cdef herr_t err_callback(void* client_data) with gil:
    # Callback which sets Python exception based on the current error stack.

    # Can't use the standard Pyrex raise because then the traceback
    # points here.  MUST be "with gil" as it can be called by nogil HDF5
    # routines.
    
    cdef H5E_error_t err_struct
    cdef H5E_major_t mj

    # Determine the error numbers for the first entry on the stack.
    H5Ewalk(H5E_WALK_UPWARD, extract_cb, &err_struct)
    mj = err_struct.maj_num

    exc = _exceptions.get(mj, H5Error)

    msg = error_string()
    PyErr_SetObject(exc, msg)

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

cdef err_c pause_errors() except? NULL:
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

# === Library init ============================================================

def _exithack():
    """ Internal function; do not call unless you want to lose all your data.
    """
    # If any identifiers have reference counts > 1 when the library closes,
    # it freaks out and dumps a message to stderr.  So we have Python dec_ref
    # everything when the interpreter's about to exit.

    cdef int count
    cdef int i
    cdef hid_t *objs

    count = H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_ALL)
    
    if count > 0:
        objs = <hid_t*>malloc(sizeof(hid_t)*count)
        try:
            H5Fget_obj_ids(H5F_OBJ_ALL, H5F_OBJ_ALL, count, objs)
            for i from 0<=i<count:
                while H5Iget_ref(objs[i]) > 1:
                    H5Idec_ref(objs[i])
        finally:
            free(objs)

cdef int import_hdf5() except -1:
    if H5open() < 0:
        raise RuntimeError("Failed to initialize the HDF5 library.")
    _enable_exceptions()
    atexit.register(_exithack)
    return 0

import_hdf5()
 
# --- Public versioning info ---

hdf5_version_tuple = get_libversion()        
hdf5_version = "%d.%d.%d" % hdf5_version_tuple
api_version_tuple = (H5PY_API_MAJ, H5PY_API_MIN)
api_version = "%d.%d" % api_version_tuple

version = H5PY_VERSION
version_tuple = []   # no list comprehensions in Pyrex
for _x in version.split('.'):
    version_tuple.append(int(_x))
version_tuple = tuple(version_tuple)

_config = H5PYConfig()








