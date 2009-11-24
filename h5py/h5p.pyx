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
    HDF5 property list interface.
"""

include "config.pxi"

# Compile-time imports
from h5 cimport init_hdf5
from utils cimport  require_tuple, convert_dims, convert_tuple, \
                    emalloc, efree, \
                    check_numpy_write, check_numpy_read
from numpy cimport ndarray, import_array
from h5t cimport TypeID, py_create

# Initialization
init_hdf5()
import_array()

# === C API ===================================================================

cdef hid_t pdefault(PropID pid):

    if pid is None:
        return <hid_t>H5P_DEFAULT
    return pid.id

cdef object propwrap(hid_t id_in):

    clsid = H5Pget_class(id_in)
    try:
        if H5Pequal(clsid, H5P_FILE_CREATE):
            pcls = PropFCID
        elif H5Pequal(clsid, H5P_FILE_ACCESS):
            pcls = PropFAID
        elif H5Pequal(clsid, H5P_DATASET_CREATE):
            pcls = PropDCID
        elif H5Pequal(clsid, H5P_DATASET_XFER):
            pcls = PropDXID
        else:
            IF H5PY_18API:
                if H5Pequal(clsid, H5P_OBJECT_COPY):
                    pcls = PropCopyID
                elif H5Pequal(clsid, H5P_LINK_CREATE):
                    pcls = PropLCID
                elif H5Pequal(clsid, H5P_LINK_ACCESS):
                    pcls = PropLAID
                elif H5Pequal(clsid, H5P_GROUP_CREATE):
                    pcls = PropGCID
                else:
                    raise ValueError("No class found for ID %d" % id_in)
            ELSE:
                raise ValueError("No class found for ID %d" % id_in)

        return pcls(id_in)
    finally:
        H5Pclose_class(clsid)

cdef object lockcls(hid_t id_in):
    cdef PropClassID pid
    pid = PropClassID(id_in)
    pid._locked = 1
    return pid


# === Public constants and data structures ====================================

# Property list classes
# These need to be locked, as the library won't let you close them.
NO_CLASS       = lockcls(H5P_NO_CLASS)
FILE_CREATE    = lockcls(H5P_FILE_CREATE)
FILE_ACCESS    = lockcls(H5P_FILE_ACCESS)
DATASET_CREATE = lockcls(H5P_DATASET_CREATE)
DATASET_XFER   = lockcls(H5P_DATASET_XFER)

IF H5PY_18API:
    OBJECT_COPY = lockcls(H5P_OBJECT_COPY)
    LINK_CREATE = lockcls(H5P_LINK_CREATE)
    LINK_ACCESS = lockcls(H5P_LINK_ACCESS)
    GROUP_CREATE = lockcls(H5P_GROUP_CREATE)

DEFAULT = None   # In the HDF5 header files this is actually 0, which is an
                 # invalid identifier.  The new strategy for default options
                 # is to make them all None, to better match the Python style
                 # for keyword arguments.


# === Property list functional API ============================================

IF H5PY_18API:
    
    def create(PropClassID cls not None):
        """(PropClassID cls) => PropID
        
        Create a new property list as an instance of a class; classes are:

        - FILE_CREATE
        - FILE_ACCESS
        - DATASET_CREATE
        - DATASET_XFER
        - LINK_CREATE
        - LINK_ACCESS
        - GROUP_CREATE
        - OBJECT_COPY
        """
        cdef hid_t newid
        newid = H5Pcreate(cls.id)
        return propwrap(newid)
ELSE:
    
    def create(PropClassID cls not None):
        """(PropClassID cls) => PropID
        
        Create a new property list as an instance of a class; classes are:

        - FILE_CREATE
        - FILE_ACCESS
        - DATASET_CREATE
        - DATASET_XFER
        """
        cdef hid_t newid
        newid = H5Pcreate(cls.id)
        return propwrap(newid)

# === Class API ===============================================================

cdef class PropID(ObjectID):

    """
        Base class for all property lists and classes
    """

    
    def equal(self, PropID plist not None):
        """(PropID plist) => BOOL

        Compare this property list (or class) to another for equality.
        """
        return <bint>(H5Pequal(self.id, plist.id))

    def __richcmp__(self, object other, int how):
        cdef bint truthval = 0
        if how != 2 and how != 3:
            return NotImplemented
        if type(self) == type(other):
            truthval = self.equal(other)
        
        if how == 2:
            return truthval
        return not truthval

    def __hash__(self):
        raise TypeError("Property lists are unhashable")

cdef class PropClassID(PropID):

    """
        An HDF5 property list class.

        * Hashable: Yes, by identifier
        * Equality: Logical H5P comparison
    """

    def __richcmp__(self, object other, int how):
        return PropID.__richcmp__(self, other, how)

    def __hash__(self):
        """ Since classes are library-created and immutable, they are uniquely
            identified by their HDF5 identifiers.
        """
        return hash(self.id)

cdef class PropInstanceID(PropID):

    """
        Base class for property list instance objects.  Provides methods which
        are common across all HDF5 property list classes.

        * Hashable: No
        * Equality: Logical H5P comparison
    """

    
    def copy(self):
        """() => PropList newid

         Create a new copy of an existing property list object.
        """
        return type(self)(H5Pcopy(self.id))

    
    def _close(self):
        """()
    
        Terminate access through this identifier.  You shouldn't have to
        do this manually, as propery lists are automatically deleted when
        their Python wrappers are freed.
        """
        H5Pclose(self.id)

    
    def get_class(self):
        """() => PropClassID

        Determine the class of a property list object.
        """
        return PropClassID(H5Pget_class(self.id))

cdef class PropCreateID(PropInstanceID):

    """
        Generic object creation property list.

        Has no methods unless HDF5 1.8.X is available.
    """
    
    IF H5PY_18API:
        pass

cdef class PropCopyID(PropInstanceID):

    """
        Generic object copy property list

        Has no methods unless HDF5 1.8.X is available
    """

    IF H5PY_18API:

        
        def set_copy_object(self, unsigned int flags):
            """(UINT flags)

            Set flags for object copying process.  Legal flags are
            from the h5o.COPY* family:

            h5o.COPY_SHALLOW_HIERARCHY_FLAG
                Copy only immediate members of a group.

            h5o.COPY_EXPAND_SOFT_LINK_FLAG
                Expand soft links into new objects.

            h5o.COPY_EXPAND_EXT_LINK_FLAG
                Expand external link into new objects.

            h5o.COPY_EXPAND_REFERENCE_FLAG
                Copy objects that are pointed to by references.

            h5o.COPY_WITHOUT_ATTR_FLAG
                Copy object without copying attributes.
            """
            H5Pset_copy_object(self.id, flags)

        
        def get_copy_object(self):
            """() => UINT flags

            Get copy process flags. Legal flags are h5o.COPY*.
            """
            cdef unsigned int flags
            H5Pget_copy_object(self.id, &flags)
            return flags


# === Concrete list implementations ===========================================

# File creation
include "h5p_fcid.pxi"

# Dataset creation
include "h5p_dcid.pxi"

# File access
include "h5p_faid.pxi"

IF H5PY_18API:

    # Link creation
    include "h5p_lcid.pxi"

    # Link access
    include "h5p_laid.pxi"

    # Group creation
    include "h5p_gcid.pxi"











