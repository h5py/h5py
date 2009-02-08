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
    Provides high-level Python objects for HDF5 files, groups, and datasets.  

    Groups provide dictionary-like access to and iteration over their members.
    File objects implicitly perform these operations on the root ('/') group.

    Datasets support Numpy-style slicing and partial I/0, including
    recarray-style access to named fields.  A minimal Numpy interface is
    included, with shape and dtype properties.

    A strong emphasis has been placed on reasonable automatic conversion from
    Python types to their HDF5 equivalents.  Setting and retrieving HDF5 data
    is almost always handled by a simple assignment.  For example, you can
    create an initialize an HDF5 dataset simply by assigning a Numpy array
    to a group:

        group["name"] = numpy.ones((10,50), dtype='<i2')

    To make it easier to get data in and out of Python, a simple command-line
    shell comes attached to all File objects via the method File.browse().
    You can explore an HDF5 file, and import datasets, groups, and named
    types into your main interactive Python session.

    Although it's not required to use this high-level interface, the full power
    of the h5py low-level library wrapping is available for these objects.
    Each highlevel object carries an identifier object (obj.id), which can be
    used by h5py.h5* functions or methods.

    It is safe to import this module using "from h5py.highlevel import *"; it
    will export only the major classes.
"""

from __future__ import with_statement

import os
import numpy
import threading
import sys

import os.path as op
import posixpath as pp

from h5py import h5, h5f, h5g, h5s, h5t, h5d, h5a, h5p, h5z, h5i
from h5py.h5 import H5Error
import h5py.selections as sel
from h5py.selections import CoordsList

import filters

config = h5.get_config()
if config.API_18:
    from h5py import h5o, h5l

__all__ = ["File", "Group", "Dataset",
           "Datatype", "AttributeManager"]

def _hbasename(name):
    """ Basename function with more readable handling of trailing slashes"""
    name = pp.basename(pp.normpath(name))
    return name if name != '' else '/'

def is_hdf5(fname):
    fname = os.path.abspath(fname)
    if os.path.isfile(fname):
        try:
            return h5f.is_hdf5(fname)
        except H5Error:
            pass
    return False

# === Base classes ============================================================

class LockableObject(object):

    """
        Base class which provides rudimentary locking support.
    """

    _lock = threading.RLock()


class HLObject(LockableObject):

    """
        Base class for high-level interface objects.

        All objects of this class support the following properties:

        id:     Low-level identifer, compatible with the h5py.h5* modules.
        name:   (Some) name of this object in the HDF5 file.
        attrs:  HDF5 attributes of this object.  See the AttributeManager docs.

        Equality comparison and hashing are based on native HDF5 object
        identity.
    """

    @property
    def name(self):
        """Name of this object in the HDF5 file.  Not necessarily unique."""
        return h5i.get_name(self.id)

    @property
    def attrs(self):
        """Provides access to HDF5 attributes. See AttributeManager."""
        return self._attrs

    def __nonzero__(self):
        return self.id.__nonzero__()

    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        if hasattr(other, 'id'):
            return self.id == other.id
        return False

class _DictCompat(object):

    """
        Contains dictionary-style compatibility methods for groups and
        attributes.
    """

    def listnames(self):
        """ Get a list containing member names """
        with self._lock:
            return list(self)

    def iternames(self):
        """ Get an iterator over member names """
        with self._lock:
            return iter(self)

    def listobjects(self):
        """ Get a list containing members """
        with self._lock:
            return [self[x] for x in self]

    def iterobjects(self):
        """ Get an iterator over members """
        with self._lock:
            for x in self:
                yield self[x]

    def listitems(self):
        """ Get a list of tuples containing (name, object) pairs """
        with self._lock:
            return [(x, self[x]) for x in self]

    def iteritems(self):
        """ Get an iterator over (name, object) pairs """
        with self._lock:
            for x in self:
                yield (x, self[x])

    def get(self, name, default):
        """ Retrieve the member, or return default if it doesn't exist """
        with self._lock:
            if name in self:
                return self[name]
            return default


class Group(HLObject, _DictCompat):

    """ Represents an HDF5 group.

        Group(parent, name, create=False)

        Group members can be accessed dictionary-style (Group["name"]).  HDF5
        objects can be automatically created in the group by assigning Numpy
        arrays, dtypes, or other Group, Dataset or Datatype objects with this
        syntax.  See the __setitem__ docstring for a complete list.

        The len() of a group is the number of members, and iterating over a
        group yields the names of its members, in arbitary library-defined
        order.  They also support the __contains__ syntax ("if name in group").

        Subgroups and datasets can be created via the convenience functions
        create_group and create_dataset, as well as by calling the appropriate
        class constructor.

        Group attributes are accessed via group.attrs; see the docstring for
        the AttributeManager class.
    """

    def __init__(self, parent_object, name, create=False):
        """ Create a new Group object, from a parent object and a name.

        If "create" is False (default), try to open the given group,
        raising an exception if it doesn't exist.  If "create" is True,
        create a new HDF5 group and link it into the parent group.
        """
        with parent_object._lock:
            if create:
                self.id = h5g.create(parent_object.id, name)
            else:
                self.id = h5g.open(parent_object.id, name)

            self._attrs = AttributeManager(self)
    
    def __setitem__(self, name, obj):
        """ Add the given object to the group.  The action taken depends on
            the type of object assigned:

        1. Named HDF5 object (Dataset, Group, Datatype):
            A hard link is created in this group which points to the
            given object.

        2. Numpy ndarray:
            The array is converted to a dataset object, with default
            settings (contiguous storage, etc.).

        3. Numpy dtype:
            Commit a copy of the datatype as a named datatype in the file.

        4. Anything else:
            Attempt to convert it to an ndarray and store it.  Scalar
            values are stored as scalar datasets. Raise ValueError if we
            can't understand the resulting array dtype.
        
        If a group member of the same name already exists, the assignment
        will fail.  You can check by using the Python __contains__ syntax:

            if "name" in grp:
                del grp["name"]
            grp["name"] = <whatever>
        """
        with self._lock:
            if isinstance(obj, Group) or isinstance(obj, Dataset) or isinstance(obj, Datatype):
                self.id.link(h5i.get_name(obj.id), name, link_type=h5g.LINK_HARD)

            elif isinstance(obj, numpy.dtype):
                htype = h5t.py_create(obj)
                htype.commit(self.id, name)

            else:
                self.create_dataset(name, data=obj)

    def __getitem__(self, name):
        """ Open an object attached to this group. 
        """
        with self._lock:
            info = h5g.get_objinfo(self.id, name)

            if info.type == h5g.DATASET:
                return Dataset(self, name)

            elif info.type == h5g.GROUP:
                return Group(self, name)

            elif info.type == h5g.TYPE:
                return Datatype(self, name)

            raise ValueError("Don't know how to open object of type %d" % info.type)

    def __delitem__(self, name):
        """ Delete (unlink) an item from this group. """
        self.id.unlink(name)

    def __len__(self):
        """ Number of members attached to this group """
        return self.id.get_num_objs()

    def __contains__(self, name):
        """ Test if a member name exists """
        return name in self.id

    def __iter__(self):
        """ Iterate over member names """
        return self.id.__iter__()

    def create_group(self, name):
        """ Create and return a subgroup.

        Fails if the group already exists.
        """
        return Group(self, name, create=True)

    def require_group(self, name):
        """ Check if a group exists, and create it if not.

        Raises H5Error if an incompatible object exists.
        """
        if not name in self:
            return self.create_group(name)
        else:
            grp = self[name]
            if not isinstance(grp, Group):
                raise H5Error("Incompatible object (%s) already exists" % grp.__class__.__name__)
            return grp

    def create_dataset(self, name, *args, **kwds):
        """ Create and return a new dataset, attached to this group.

        create_dataset(name, shape, [dtype=<Numpy dtype>], **kwds)
        create_dataset(name, data=<Numpy array>, **kwds)

        If "dtype" is not specified, the default is single-precision
        floating point, with native byte order ("=f4").

        Creating a dataset will fail if another of the same name already 
        exists. Additional keywords are:

        chunks:        Tuple of chunk dimensions or None*
        compression:   DEFLATE (gzip) compression level, int or None*
        shuffle:       Use the shuffle filter? (requires compression) T/F*
        fletcher32:    Enable Fletcher32 error detection? T/F*
        maxshape:      Tuple giving dataset maximum dimensions or None*.
                       You can grow each axis up to this limit using
                       resize().  For each unlimited axis, provide None.

        All these options require chunking.  If a chunk tuple is not
        provided, the constructor will guess an appropriate chunk shape.
        Please note none of these are allowed for scalar datasets.
        """
        return Dataset(self, name, *args, **kwds)

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        """Open a dataset, or create it if it doesn't exist.

        Checks if a dataset with compatible shape and dtype exists, and
        creates one if it doesn't.  Raises H5Error if an incompatible
        dataset (or group) already exists.  

        By default, datatypes are compared for loss-of-precision only.
        To require an exact match, set keyword "exact" to True.  Shapes
        are always compared exactly.

        Keyword arguments are only used when creating a new dataset; they
        are ignored if an dataset with matching shape and dtype already
        exists.  See create_dataset for a list of legal keywords.
        """
        dtype = numpy.dtype(dtype)

        with self._lock:
            if not name in self:
                return self.create_dataset(name, *(shape, dtype), **kwds)

            dset = self[name]
            if not isinstance(dset, Dataset):
                raise H5Error("Incompatible object (%s) already exists" % dset.__class__.__name__)

            if not shape == dset.shape:
                raise H5Error("Shapes do not match (existing %s vs new %s)" % (dset.shape, shape))

            if exact:
                if not dtype == dset.dtype:
                    raise H5Error("Datatypes do not exactly match (existing %s vs new %s)" % (dset.dtype, dtype))
            elif not numpy.can_cast(dtype, dset.dtype):
                raise H5Error("Datatypes cannot be safely cast (existing %s vs new %s)" % (dset.dtype, dtype))
            
            return dset


    # New 1.8.X methods

    def copy(self, source, dest, name=None):
        """ Copy an object or group.

        The source can be a path, Group, Dataset, or Datatype object.  The
        destination can be either a path or a Group object.  The source and
        destinations need not be in the same file.

        When the destination is a Group object, by default the target will
        be created in that group with its current name (basename of obj.name).
        You can override that by setting "name" to a string.

        Example:

        >>> f = File('myfile.hdf5')
        >>> f.listnames()
        ['MyGroup']
        >>> f.copy('MyGroup', 'MyCopy')
        >>> f.listnames()
        ['MyGroup', 'MyCopy']

        """
        if not config.API_18:
            raise NotImplementedError("This feature is only available with HDF5 1.8.0 and later")

        with self._lock:

            if isinstance(source, HLObject):
                source_path = '.'
            else:
                # Interpret source as a path relative to this group
                source_path = source
                source = self

            if isinstance(dest, Group):
                if name is not None:
                    dest_path = name
                else:
                    dest_path = pp.basename(h5i.get_name(source[source_path].id))

            elif isinstance(dest, HLObject):
                raise TypeError("Destination must be path or Group object")
            else:
                # Interpret destination as a path relative to this group
                dest_path = dest
                dest = self

            h5o.copy(source.id, source_path, dest.id, dest_path)

    def visit(self, func):
        """ Recursively visit all names in this group and subgroups.

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:

            func(<member name>) => <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guranteed.

        Example:

        >>> # List the entire contents of the file
        >>> f = File("foo.hdf5")
        >>> list_of_names = []
        >>> f.visit(list_of_names.append)

        Only available with HDF5 1.8.X.
        """
        if not config.API_18:
            raise NotImplementedError("This feature is only available with HDF5 1.8.0 and later")
    
        with self._lock:
            return h5o.visit(self.id, func)

    def visititems(self, func):
        """ Recursively visit names and objects in this group and subgroups.

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:

            func(<member name>, <object>) => <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guranteed.

        Example:

        # Get a list of all datasets in the file
        >>> mylist = []
        >>> def func(name, obj):
        ...     if isinstance(obj, Dataset):
        ...         mylist.append(name)
        ...
        >>> f = File('foo.hdf5')
        >>> f.visititems(func)

        Only available with HDF5 1.8.X.
        """
        if not config.API_18:
            raise NotImplementedError("This feature is only available with HDF5 1.8.0 and later")

        with self._lock:
            def call_proxy(name):
                return func(name, self[name])
            return h5o.visit(self.id, call_proxy)

    def __repr__(self):
        with self._lock:
            try:
                return '<HDF5 group "%s" (%d members)>' % \
                    (_hbasename(self.name), len(self))
            except Exception:
                return "<Closed HDF5 group>"

class File(Group):

    """ Represents an HDF5 file on disk.

        File(name, mode='a')

        Legal modes: r, r+, w, w-, a (default)

        File objects inherit from Group objects; Group-like methods all
        operate on the HDF5 root group ('/').  Like Python file objects, you
        must close the file ("obj.close()") when you're done with it.

        The special method browse() will open a command shell, allowing you
        to browse the file and import objects into the interactive Python
        session.  If the readline module is available, this includes things
        like command history and tab completion.

        This object supports the Python context manager protocol, when used
        in a "with" block::

            with File(...) as f:
                ... do stuff with f...
            # end block
       
        The file will be closed at the end of the block, regardless of any
        exceptions raised. 
    """

    @property
    def name(self):
        """File name on disk"""
        return self._name

    @property
    def mode(self):
        """Python mode used to open file"""
        return self._mode

    # --- Public interface (File) ---------------------------------------------

    def __init__(self, name, mode='a'):
        """ Create a new file object.  

            Valid modes (like Python's file() modes) are: 
            - r   Readonly, file must exist
            - r+  Read/write, file must exist
            - w   Create file, truncate if exists
            - w-  Create file, fail if exists
            - a   Read/write if exists, create otherwise (default)
        """
        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_fclose_degree(h5f.CLOSE_STRONG)
        if mode == 'r':
            self.fid = h5f.open(name, h5f.ACC_RDONLY, fapl=plist)
        elif mode == 'r+':
            self.fid = h5f.open(name, h5f.ACC_RDWR, fapl=plist)
        elif mode == 'w-':
            self.fid = h5f.create(name, h5f.ACC_EXCL, fapl=plist)
        elif mode == 'w':
            self.fid = h5f.create(name, h5f.ACC_TRUNC, fapl=plist)
        elif mode == 'a':
            if not os.path.exists(name):
                self.fid = h5f.create(name, h5f.ACC_EXCL, fapl=plist)
            else:
                self.fid = h5f.open(name, h5f.ACC_RDWR, fapl=plist)
        else:
            raise ValueError("Invalid mode; must be one of r, r+, w, w-, a")

        self.id = self.fid  # So the Group constructor can find it.
        Group.__init__(self, self, '/')

        self._name = name
        self._mode = mode

    def close(self):
        """ Close this HDF5 file.  All open objects will be invalidated.
        """
        with self._lock:
            self.id._close()
            self.fid.close()

    def flush(self):
        """ Tell the HDF5 library to flush its buffers.
        """
        h5f.flush(self.fid)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        with self._lock:
            if self.id._valid:
                self.close()
            
    def __repr__(self):
        with self._lock:
            try:
                return '<HDF5 file "%s" (mode %s, %d root members)>' % \
                    (os.path.basename(self.name), self.mode, len(self))
            except Exception:
                return "<Closed HDF5 file>"

    # Fix up identity to use the file identifier, not the root group.
    def __hash__(self):
        return hash(self.fid)
    def __eq__(self, other):
        if hasattr(other, 'fid'):
            return self.fid == other.fid
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

class Dataset(HLObject):

    """ High-level interface to an HDF5 dataset.

        Dataset(group, name, shape=None, dtype=None, data=None, **kwds)

        Datasets behave superficially like Numpy arrays.  The full Numpy
        slicing syntax, including recarray indexing of named fields (even
        more than one), is supported.  The object returned is always a
        Numpy ndarray.

        Among others, the following properties are provided:
          shape:    Numpy-style shape tuple of dimsensions
          dtype:    Numpy dtype representing the datatype
          value:    Copy of the full dataset, as either a Numpy array or a
                     Numpy/Python scalar, depending on the shape.
    """

    def _g_shape(self):
        """Numpy-style shape tuple giving dataset dimensions"""
        return self.id.shape

    def _s_shape(self, shape):
        self.resize(shape)

    shape = property(_g_shape, _s_shape)

    @property
    def dtype(self):
        """Numpy dtype representing the datatype"""
        return self.id.dtype

    @property
    def value(self):
        """The entire dataset, as an array or scalar depending on the shape"""
        with self._lock:
            arr = self[...]
            #if arr.shape == ():
            #    return numpy.asscalar(arr)
            return arr

    @property
    def chunks(self):
        """Dataset chunks (or None)"""
        return self._chunks

    @property
    def compression(self):
        """Compression strategy (or None)"""
        for x in ('gzip','lzf','szip'):
            if x in self._filters:
                return x
        return None

    @property
    def compression_opts(self):
        """ Compression setting.  Int(0-9) for gzip, 2-tuple for szip. """
        return self._filters.get(self.compression, None)

    @property
    def shuffle(self):
        """Shuffle filter present (T/F)"""
        return 'shuffle' in self._filters

    @property
    def fletcher32(self):
        """Fletcher32 filter is present (T/F)"""
        return 'fletcher32' in self._filters
        
    @property
    def maxshape(self):
        space = self.id.get_space()
        dims = space.get_simple_extent_dims(True)
        return tuple(x if x != h5s.UNLIMITED else None for x in dims)

    def __init__(self, group, name,
                    shape=None, dtype=None, data=None,
                    chunks=None, compression=None, shuffle=None,
                    fletcher32=None, maxshape=None, compression_opts=None):
        """ Open or create a new dataset in the file.

        It's recommended you use the Group methods (open via Group["name"],
        create via Group.create_dataset), rather than calling the constructor.

        There are two modes of operation for this constructor:

        1.  Open an existing dataset:
              Dataset(group, name)

        2.  Create a dataset:
              Dataset(group, name, shape, [dtype=<Numpy dtype>], **kwds)
            or
              Dataset(group, name, data=<Numpy array>, **kwds)

              If "dtype" is not specified, the default is single-precision
              floating point, with native byte order ("=f4").

        Creating a dataset will fail if another of the same name already 
        exists.  Also, chunks/compression/shuffle/fletcher32 may only be
        specified when creating a dataset.

        Creation keywords (* is default):

        chunks:        Tuple of chunk dimensions, True, or None*
        compression:   "gzip", "lzf", or "szip" (if available)
        shuffle:       Use the shuffle filter? (requires compression) T/F*
        fletcher32:    Enable Fletcher32 error detection? T/F*
        maxshape:      Tuple giving dataset maximum dimensions or None*.
                       You can grow each axis up to this limit using
                       resize().  For each unlimited axis, provide None.
        
        compress_opts: Optional setting for the compression filter

        All these options require chunking.  If a chunk tuple is not
        provided, the constructor will guess an appropriate chunk shape.
        Please note none of these are allowed for scalar datasets.
        """
        with group._lock:
            if data is None and shape is None:
                if any((dtype,chunks,compression,shuffle,fletcher32)):
                    raise ValueError('You cannot specify keywords when opening a dataset.')
                self.id = h5d.open(group.id, name)
            else:
                
                # Convert data to a C-contiguous ndarray
                if data is not None:
                    data = numpy.asarray(data, order="C")

                # Validate shape
                if shape is None:
                    if data is None:
                        raise TypeError("Either data or shape must be specified")
                    shape = data.shape
                else:
                    shape = tuple(shape)
                    if data is not None and (numpy.product(shape) != numpy.product(data.shape)):
                        raise ValueError("Shape tuple is incompatible with data")

                # Validate dtype
                if dtype is None and data is None:
                    dtype = numpy.dtype("=f4")
                elif dtype is None and data is not None:
                    dtype = data.dtype
                else:
                    dtype = numpy.dtype(dtype)

                if dtype.subdtype is not None:
                    raise TypeError("ARRAY types are only supported as members of a compound type")

                # Legacy
                if any((compression, shuffle, fletcher32, maxshape)):
                    if chunks is False:
                        raise ValueError("Chunked format required for given storage options")

                # Legacy
                if compression in range(10) or compression is True:
                    if compression_opts is None:
                        if compression is True:
                            compression_opts = 4
                        else:
                            compression_opts = compression
                    else:
                        raise TypeError("Conflict in compression options")
                    compression = 'gzip'

                # Generate the dataset creation property list
                # This also validates the keyword arguments
                plist = filters.generate_dcpl(shape, dtype, chunks, compression,
                            compression_opts, shuffle, fletcher32, maxshape)

                if maxshape is not None:
                    maxshape = tuple(x if x is not None else h5s.UNLIMITED for x in maxshape)

                space_id = h5s.create_simple(shape, maxshape)
                type_id = h5t.py_create(dtype)

                self.id = h5d.create(group.id, name, type_id, space_id, plist)
                if data is not None:
                    self.id.write(h5s.ALL, h5s.ALL, data)

            self._attrs = AttributeManager(self)
            plist = self.id.get_create_plist()
            self._filters = filters.get_filters(plist)
            if plist.get_layout() == h5d.CHUNKED:
                self._chunks = plist.get_chunk()
            else:
                self._chunks = None

    def resize(self, size, axis=None):
        """ Resize the dataset, or the specified axis.

        The dataset must be stored in chunked format.

        Argument should be either a new shape tuple, or an integer.  The rank
        of the dataset cannot be changed.  Keep in mind the dataset can only
        be resized up to the maximum dimensions provided when it was created.

        Beware; if the array has more than one dimension, the indices of
        existing data can change.

        Only available with HDF5 1.8.
        """
        with self._lock:

            if not config.API_18:
                raise NotImplementedError("Resizing is only available with HDF5 1.8.")

            if self.chunks is None:
                raise TypeError("Only chunked datasets can be resized")

            if axis is not None:
                if not axis >=0 and axis < self.id.rank:
                    raise ValueError("Invalid axis (0 to %s allowed)" % self.id.rank-1)
                try:
                    newlen = int(size)
                except TypeError:
                    raise TypeError("Argument must be a single int if axis is specified")
                size = list(self.shape)
                size[axis] = newlen

            size = tuple(size)
            self.id.set_extent(size)
            h5f.flush(self.id)  # THG recommends
            
    def __len__(self):
        """ The size of the first axis.  TypeError if scalar.
        """
        size = self.len()
        if size > sys.maxint:
            raise OverflowError("Value too big for Python's __len__; use Dataset.len() instead.")
        return size

    def len(self):
        """ The size of the first axis.  TypeError if scalar. 

            Use of this method is preferred to len(dset), as Python's built-in
            len() cannot handle values greater then 2**32 on 32-bit systems.
        """
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Attempt to take len() of scalar dataset")
        return shape[0]

    def __iter__(self):
        """ Iterate over the first axis.  TypeError if scalar.  Modifications
            to the yielded data are *NOT* recorded.
        """
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Can't iterate over a scalar dataset")
        for i in xrange(shape[0]):
            yield self[i]

    def __getitem__(self, args):
        """ Read a slice from the HDF5 dataset.

        Takes slices and recarray-style field names (more than one is
        allowed!) in any order.  Obeys basic NumPy rules, including
        broadcasting.

        Also supports:

        * Boolean "mask" array indexing
        * Advanced dataspace selection via the "selections" module
        """
        with self._lock:

            args = args if isinstance(args, tuple) else (args,)

            # Sort field indices from the rest of the args.
            names = tuple(x for x in args if isinstance(x, str))
            args = tuple(x for x in args if not isinstance(x, str))

            # Create NumPy datatype for read, using only the named fields
            # as specified by the user.
            basetype = self.id.dtype
            if len(names) == 0:
                new_dtype = basetype
            else:
                for name in names:
                    if not name in basetype.names:
                        raise ValueError("Field %s does not appear in this type." % name)
                new_dtype = numpy.dtype([(name, basetype.fields[name][0]) for name in names])

            # Perform the dataspace selection.
            selection = sel.select(self.shape, args)

            if selection.nselect == 0:
                return numpy.ndarray((0,), dtype=new_dtype)

            # Create the output array using information from the selection.
            arr = numpy.ndarray(selection.mshape, new_dtype, order='C')

            # Perfom the actual read
            mspace = h5s.create_simple(selection.mshape)
            fspace = selection._id
            self.id.read(mspace, fspace, arr)

            # Patch up the output for NumPy
            if len(names) == 1:
                arr = arr[names[0]]     # Single-field recarray convention
            if arr.shape == ():
                arr = numpy.asscalar(arr)
            return arr

    def __setitem__(self, args, val):
        """ Write to the HDF5 dataset from a Numpy array.

        NumPy's broadcasting rules are honored, for "simple" indexing
        (slices and integers).  For advanced indexing, the shapes must
        match.

        Classes from the "selections" module may also be used to index.s
        """
        with self._lock:

            args = args if isinstance(args, tuple) else (args,)

            # Sort field indices from the slicing
            names = tuple(x for x in args if isinstance(x, str))
            args = tuple(x for x in args if not isinstance(x, str))

            if len(names) != 0:
                raise TypeError("Field name selections are not allowed for write.")

            # 3. Validate the input array
            val = numpy.asarray(val, order='C')

            # 4. Perform the dataspace selection
            selection = sel.select(self.shape, args)

            if selection.nselect == 0:
                return

            # 5. Broadcast scalars if necessary
            if val.shape == () and selection.mshape != ():
                val2 = numpy.empty(selection.mshape[-1], dtype=val.dtype)
                val2[...] = val
                val = val2
            
            # 6. Perform the write, with broadcasting
            mspace = h5s.create_simple(val.shape, (h5s.UNLIMITED,)*len(val.shape))
            for fspace in selection.broadcast(val.shape):
                self.id.write(mspace, fspace, val)

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        """ Read data directly from HDF5 into an existing NumPy array.

        The destination array must be C-contiguous and writable.
        Selections may be any operator class (HyperSelection, etc) in
        h5py.selections, or the output of numpy.s_[<args>].

        Broadcasting is supported for simple indexing.
        """

        if source_sel is None:
            source_sel = sel.SimpleSelection(self.shape)
        else:
            source_sel = sel.select(self.shape, source_sel)  # for numpy.s_
        fspace = source_sel._id

        if dest_sel is None:
            dest_sel = sel.SimpleSelection(dest.shape)
        else:
            dest_sel = sel.select(dest.shape, dest_sel)

        for mspace in dest_sel.broadcast(source_sel.mshape):
            self.id.read(mspace, fspace, dest)

    def write_direct(self, source, source_sel=None, dest_sel=None):
        """ Write data directly to HDF5 from a NumPy array.

        The source array must be C-contiguous.  Selections may be any
        operator class (HyperSelection, etc) in h5py.selections, or
        the output of numpy.s_[<args>].

        Broadcasting is supported for simple indexing.
        """

        if source_sel is None:
            source_sel = sel.SimpleSelection(source.shape)
        else:
            source_sel = sel.select(source.shape, source_sel)  # for numpy.s_
        mspace = source_sel._id

        if dest_sel is None:
            dest_sel = sel.SimpleSelection(self.shape)
        else:
            dest_sel = sel.select(self.shape, dest_sel)

        for fspace in dest_sel.broadcast(source_sel.mshape):
            self.id.write(mspace, fspace, source)

    def __repr__(self):
        with self._lock:
            try:
                return '<HDF5 dataset "%s": shape %s, type "%s">' % \
                    (_hbasename(self.name), self.shape, self.dtype.str)
            except Exception:
                return "<Closed HDF5 dataset>"

class AttributeManager(LockableObject, _DictCompat):

    """ Allows dictionary-style access to an HDF5 object's attributes.

        These come attached to HDF5 objects as <obj>.attrs.  There's no need to
        create one yourself.

        Like the members of groups, attributes are accessed using dict-style
        syntax.  Anything which can be reasonably converted to a Numpy array or
        Numpy scalar can be stored.

        Since attributes are typically used for small scalar values, acessing
        a scalar attribute returns a Numpy/Python scalar, not an 0-dimensional
        array.  Non-scalar data is always returned as an ndarray.

        The len() of this object is the number of attributes; iterating over
        it yields the attribute names.  They also support the __contains__
        syntax ("if name in obj.attrs...").

        Unlike groups, writing to an attribute will overwrite an existing
        attribute of the same name.  This is not a transacted operation; you
        can lose data if you try to assign an object which h5py doesn't
        understand.
    """

    def __init__(self, parent):
        """ Private constructor; you should not create these.
        """
        self.id = parent.id

    def __getitem__(self, name):
        """ Read the value of an attribute.
    
        If the attribute is scalar, it will be returned as a Numpy
        scalar.  Otherwise, it will be returned as a Numpy ndarray.
        """
        with self._lock:
            attr = h5a.open(self.id, name)

            arr = numpy.ndarray(attr.shape, dtype=attr.dtype, order='C')
            attr.read(arr)

            if len(arr.shape) == 0:
                return numpy.asscalar(arr)
            return arr

    def __setitem__(self, name, value):
        """ Set the value of an attribute, overwriting any previous value.

        The value you provide must be convertible to a Numpy array or scalar.

        Any existing value is destroyed just before the call to h5a.create.
        If the creation fails, the data is not recoverable.
        """
        with self._lock:
            value = numpy.asarray(value, order='C')

            space = h5s.create_simple(value.shape)
            htype = h5t.py_create(value.dtype)

            # TODO: some kind of transactions safeguard
            if name in self:
                h5a.delete(self.id, name)

            attr = h5a.create(self.id, name, htype, space)
            attr.write(value)

    def __delitem__(self, name):
        """ Delete an attribute (which must already exist). """
        h5a.delete(self.id, name)

    def __len__(self):
        """ Number of attributes attached to the object. """
        # I expect we will not have more than 2**32 attributes
        return h5a.get_num_attrs(self.id)

    def __iter__(self):
        """ Iterate over the names of attributes. """
        with self._lock:
            attrlist = []
            def iter_cb(name, *args):
                attrlist.append(name)
            h5a.iterate(self.id, iter_cb)

            for name in attrlist:
                yield name

    def __contains__(self, name):
        """ Determine if an attribute exists, by name. """
        return h5a.exists(self.id, name)

    def __repr__(self):
        with self._lock:
            try:
                return '<Attributes of HDF5 object "%s" (%d)>' % \
                    (_hbasename(h5i.get_name(self.id)), len(self))
            except Exception:
                return "<Attributes of closed HDF5 object>"


class Datatype(HLObject):

    """
        Represents an HDF5 named datatype.

        These intentionally only represent named types, and exist mainly so
        that you can access their attributes.  The property Datatype.dtype
        provides a Numpy dtype equivalent.

        They're produced only by indexing into a Group object; you can't create
        one manually.  To create a named datatype in a file:

            group["name"] = <Numpy dtype object> | <Datatype object>*

            * will create hard link to an existing type
    """

    @property
    def dtype(self):
        """Numpy dtype equivalent for this datatype"""
        return self.id.dtype

    def __init__(self, grp, name):
        """ Private constructor; you should not create these.
        """
        with grp._lock:
            self.id = h5t.open(grp.id, name)
            self._attrs = AttributeManager(self)

    def __repr__(self):
        with self._lock:
            try:
                return '<HDF5 named type "%s" (dtype %s)>' % \
                    (_hbasename(self.name), self.dtype.str)
            except Exception:
                return "<Closed HDF5 named type>"




