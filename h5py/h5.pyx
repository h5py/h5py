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
__doc__ = \
"""
    Common support and versioning module for the h5py HDF5 interface.

    This is an internal module which is designed to set up the library and
    enables HDF5 exception handling.  It also enables debug logging, if the
    library has been compiled with a nonzero debugging level.

    All exception classes and error handling functions are also in this module.
"""

include "config.pxi"

from python_exc cimport PyErr_SetString

import atexit
import threading

IF H5PY_18API:
    ITER_INC    = H5_ITER_INC     # Increasing order
    ITER_DEC    = H5_ITER_DEC     # Decreasing order
    ITER_NATIVE = H5_ITER_NATIVE  # No particular order, whatever is fastest

    INDEX_NAME      = H5_INDEX_NAME       # Index on names      
    INDEX_CRT_ORDER = H5_INDEX_CRT_ORDER  # Index on creation order    

cdef class SmartStruct:

    """ Provides basic mechanics for structs """
    
    def _hash(self):
        raise TypeError("%s instances are unhashable" % self.__class__.__name__)

    def __hash__(self):
        # This is forwarded so that I don't have to reimplement __richcmp__ everywhere
        return self._hash()

    def __richcmp__(self, object other, int how):
        """Equality based on hash.  If unhashable, NotImplemented."""
        cdef bint truthval = 0

        if how != 2 and how != 3:
            return NotImplemented

        if isinstance(other, type(self)):
            try:
                truthval = hash(self) == hash(other)
            except TypeError:
                return NotImplemented

        if how == 2:
            return truthval
        return not truthval

    def __str__(self):
        """Format "name: value" pairs recursively for public attributes"""
        mems = dict([(x, str(getattr(self, x))) for x in dir(self) if not x.startswith('_')])
        for x in mems:
            if isinstance(getattr(self,x), SmartStruct):
                mems[x] = "\n"+"\n".join(["    "+y for y in mems[x].split("\n")[1:]])
        hdr = "=== %s ===\n" % self.__class__.__name__ if self._title is None else self._title
        return hdr+"\n".join(["%s: %s" % (name, mems[name]) for name in sorted(mems)])

cdef class H5PYConfig:

    """
        Provides runtime access to global library settings.
    """

    def __init__(self):
        self.API_16 = H5PY_16API
        self.API_18 = H5PY_18API
        self.DEBUG = H5PY_DEBUG
        self._r_name = 'r'
        self._i_name = 'i'
        self._f_name = 'FALSE'
        self._t_name = 'TRUE'

    property complex_names:
        """ Settable 2-tuple controlling how complex numbers are saved.

        Format is (real_name, imag_name), defaulting to ('r','i').
        """

        def __get__(self):
            return (self._r_name, self._i_name)

        def __set__(self, val):
            try:
                r = str(val[0])
                i = str(val[1])
            except Exception:
                raise TypeError("complex_names must be a 2-tuple of strings (real, img)")
            self._r_name = r
            self._i_name = i

    property bool_names:
        """ Settable 2-tuple controlling HDF5 ENUM names for boolean types.

        Format is (false_name, real_name), defaulting to ('FALSE', 'TRUE').
        """
        def __get__(self):
            return (self._f_name, self._t_name)

        def __set__(self, val):
            try:
                f = str(val[0])
                t = str(val[1])
            except Exception:
                raise TypeError("bool_names must be a 2-tuple of strings (false, true)")
            self._f_name = f
            self._t_name = t

    def __repr__(self):
        rstr =  \
"""\
Summary of h5py config
======================
HDF5: %s
1.6 API: %s
1.8 API: %s
Diagnostic mode: %s
Complex names: %s"""

        rstr %= ("%d.%d.%d" % get_libversion(), bool(self.API_16),
                bool(self.API_18), bool(self.DEBUG),
                self.complex_names)
        return rstr

cdef H5PYConfig cfg = H5PYConfig()

cpdef H5PYConfig get_config():
    """() => H5PYConfig

    Get a reference to the global library configuration object
    """
    return cfg

# === Bootstrap diagnostics and threading, before decorator is defined ===

IF H5PY_DEBUG:
    import logging
    for x in ('h5py.library', 'h5py.identifiers', 'h5py.functions', 'h5py.threads'):
        l = logging.getLogger(x)
        l.setLevel(H5PY_DEBUG)
        l.addHandler(logging.StreamHandler())
    log_lib = logging.getLogger('h5py.library')
    log_ident = logging.getLogger('h5py.identifiers')
    log_threads = logging.getLogger('h5py.threads')


def loglevel(lev):
    """ (INT lev)
        
        Shortcut to set the logging level on all library streams.
        Does nothing if not built in debug mode.
    """
    IF H5PY_DEBUG:
        for x in ('h5py.identifiers', 'h5py.functions', 'h5py.threads'):
            l = logging.getLogger(x)
            l.setLevel(lev)
    ELSE:
        pass

cdef class PHIL:

    """
        The Primary HDF5 Interface Lock (PHIL) is a global reentrant lock
        which manages access to the library.  HDF5 is not guaranteed to 
        be thread-safe, and certain callbacks in h5py can execute arbitrary
        threaded Python code, defeating the normal GIL-based protection for
        extension modules.  Therefore, in all routines acquire this lock first.

        You should NOT use this object in your code.  It's internal to the
        library.
    """

    def __init__(self):
        self.lock = threading.RLock()
    cpdef bint __enter__(self) except -1:
        self.lock.acquire()
        return 0
    cpdef bint __exit__(self,a,b,c) except -1:
        self.lock.release()
        return 0
    cpdef bint acquire(self, int blocking=1) except -1:
        register_thread()
        cdef bint rval = self.lock.acquire(blocking)
        return rval
    cpdef bint release(self) except -1:
        self.lock.release()
        return 0   

cdef PHIL phil = PHIL()

cpdef PHIL get_phil():
    """() => PHIL

    Obtain a reference to the PHIL.
    """
    global phil
    return phil

# Everything required for the decorator is now defined

from _sync import sync, nosync

# === Public C API for object identifiers =====================================

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
            phil.acquire()
            try:
                return H5Iget_type(self.id) != H5I_BADID
            finally:
                phil.release()
    
    def __nonzero__(self):
        """ Truth value for object identifiers (like _valid) """
        return self._valid

    def __cinit__(self, hid_t id_):
        """ Object init; simply records the given ID. """
        self._locked = 0
        self.id = id_

    IF H5PY_DEBUG:
        def __init__(self, hid_t id_):
            log_ident.debug("+ %s" % str(self))

    def __dealloc__(self):
        """ Automatically decrefs the ID, if it's valid. """

        # Acquiring PHIL leads to segfault in presence of cyclic
        # garbage collection.  We'll have to hope this isn't called while
        # an HDF5 callback is in progress.

        IF H5PY_DEBUG:
            log_ident.debug("- %d" % self.id)
        if (not self._locked) and H5Iget_type(self.id) != H5I_BADID:
            H5Idec_ref(self.id)

    
    def __copy__(self):
        """ Create another object wrapper which points to the same id. 

            WARNING: Locks (i.e. datatype lock() methods) do NOT work correctly
            across copies.
        """
        cdef ObjectID copy
        phil.acquire()
        try:
            copy = type(self)(self.id)
            if self._valid and not self._locked:
                H5Iinc_ref(self.id)
            copy._locked = self._locked
            IF H5PY_DEBUG:
                log_ident.debug("c %s" % str(self))
            return copy
        finally:
            phil.release()

    def __richcmp__(self, object other, int how):
        """ Basic comparison for HDF5 objects.  Implements only equality:

            1. Mismatched types always NOT EQUAL
            2. Try to compare object hashes
            3. If unhashable, compare identifiers
        """
        cdef bint truthval = 0

        if how != 2 and how != 3:
            return NotImplemented

        if isinstance(other, type(self)):
            try:
                truthval = hash(self) == hash(other)
            except TypeError:
                truthval = self.id == other.id

        if how == 2:
            return truthval
        return not truthval

    def __hash__(self):
        """ Default hash is computed from the object header, which requires
            a file-resident object.  TypeError if this can't be done.
        """
        cdef H5G_stat_t stat

        if self._hash is None:
            phil.acquire()
            try:
                H5Gget_objinfo(self.id, '.', 0, &stat)
                self._hash = hash((stat.fileno[0], stat.fileno[1], stat.objno[0], stat.objno[1]))
            except Exception:
                raise TypeError("Objects of class %s cannot be hashed" % self.__class__.__name__)
            finally:
                phil.release()

        return self._hash

    def __repr__(self):
        phil.acquire()
        try:
            ref = str(H5Iget_ref(self.id)) if self._valid else "X"
            lck = "L" if self._locked else "U"
            return "%s [%s] (%s) %d" % (self.__class__.__name__, ref, lck, self.id)
        finally:
            phil.release()


# === HDF5 "H5" API ===========================================================

@sync
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

@sync
def _close():
    """ Internal function; do not call unless you want to lose all your data.
    """
    H5close()

@sync
def _open():
    """ Internal function; do not call unless you want to lose all your data.
    """
    H5open()


# === Public exception hierarchy ==============================================

class H5Error(Exception):
    """ Base class for internal HDF5 library exceptions.
    """
    pass

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

cdef dict _exceptions = {
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

        Attributes:

        * maj_num:    INT major error number
        * min_num:    INT minor error number
        * func_name:  STRING name of failing function
        * file_name:  STRING name of file in which error occurreed
        * line:       UINT line number at which error occured
        * desc:       STRING description of error
    """
    cdef readonly int maj_num
    cdef readonly int min_num
    cdef readonly object func_name
    cdef readonly object file_name
    cdef readonly unsigned int line
    cdef readonly object desc

    @sync
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

@sync
def error_stack():
    """ () => LIST error_stack

        Retrieve the HDF5 error stack as a list of ErrorStackElement objects,
        with the most recent call (the deepest one) listed last.
    """
    stack = []
    H5Ewalk(H5E_WALK_DOWNWARD, walk_cb, <void*>stack)
    return stack

cpdef object error_string():
    """ () => STRING error_stack

        Return a string representation of the current error condition.
        Format is one line of the format::

            '<Description> (<Function name>: <error type>)'

        If the stack is more than one level deep, this is followed by a
        header and then n lines of the format::

            '    n: "<Description>" at <function name>'
    """
    cdef int stacklen
    cdef ErrorStackElement el

    stack = error_stack()
    stacklen = len(stack)

    if stacklen == 0:
        msg = "No HDF5 error recorded"
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

@sync
def clear():
    """ ()

        Clear the error stack.
    """
    H5Eclear()

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

    # MUST be "with gil" as it can be called by nogil HDF5 routines.
    # By definition any function for which this can be called already
    # holds the PHIL.
    
    cdef H5E_error_t err_struct
    cdef H5E_major_t mj

    # Determine the error numbers for the first entry on the stack.
    H5Ewalk(H5E_WALK_UPWARD, extract_cb, &err_struct)
    mj = err_struct.maj_num

    try:
        exc = _exceptions[mj]
    except KeyError:
        exc = H5Error

    msg = error_string()
    PyErr_SetString(exc, msg)  # Can't use "raise" or the traceback points here

    return 1


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
    
    IF H5PY_DEBUG:
        log_lib.info("* h5py is shutting down (closing %d leftover IDs)" % count)

    if count > 0:
        objs = <hid_t*>malloc(sizeof(hid_t)*count)
        try:
            H5Fget_obj_ids(H5F_OBJ_ALL, H5F_OBJ_ALL, count, objs)
            for i from 0<=i<count:
                while H5Iget_type(objs[i]) != H5I_BADID and H5Iget_ref(objs[i]) > 0:
                    H5Idec_ref(objs[i])
        finally:
            free(objs)

hdf5_inited = 0

cpdef int register_thread() except -1:
    """ Register the current thread for native HDF5 exception support.
    """
    if H5Eset_auto(err_callback, NULL) < 0:
        raise RuntimeError("Failed to register HDF5 exception callback")
    return 0

cdef int init_hdf5() except -1:
    # Initialize the library and register Python callbacks for exception
    # handling.  Safe to call more than once.
    global hdf5_inited

    if not hdf5_inited:
        IF H5PY_DEBUG:
            log_lib.info("* Initializing h5py library")
        if H5open() < 0:
            raise RuntimeError("Failed to initialize the HDF5 library.")
        register_thread()
        if register_lzf() < 0:
            raise RuntimeError("Failed to register LZF filter")
        atexit.register(_exithack)
        hdf5_inited = 1

    return 0

init_hdf5()
 
# === Module init =============================================================

_hdf5_version_tuple = get_libversion()        
_api_version_tuple = (int(H5PY_API/10), H5PY_API%10)
_version_tuple = tuple([int(x) for x in H5PY_VERSION.split('.')])






