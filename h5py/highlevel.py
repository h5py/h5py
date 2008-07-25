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

    Datasets support full Numpy-style slicing and partial I/0, including
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
import inspect
import threading
from weakref import WeakValueDictionary

from h5py import h5, h5f, h5g, h5s, h5t, h5d, h5a, h5p, h5z, h5i
from h5py.h5 import H5Error
from utils_hl import slicer, hbasename, strhdr, strlist
from browse import _H5Browser


__all__ = ["LockableObject", "HLObject", "File", "Group", "Dataset",
           "Datatype", "AttributeManager"]

try:
    # For interactive File.browse() capability
    import readline
except ImportError:
    readline = None

class LockableObject(object):

    """
        Base class which provides rudimentary locking support.
    """

    __locks = WeakValueDictionary()   # Key => RLock object
    __locks_lock = threading.RLock()

    def _get_lock(self):
        """ Get an reentrant lock object appropriate for this object.

            Returns the same lock for each unique underlying HDF5 object:
             1. For named objects, use fileno/objno as key (guaranteed unique)
             2. For transient objects, use the HDF5 integer identifier

            This has the following limitations:
             1. File objects can be locked, but this is not very useful because
                there's no obvious way to represent the dependency relationship
                between files and the objects they contain.

             2. In cases where different transient identifiers refer to the
                same object, it will not be properly locked.  Currently no
                high-level objects are transient.

            Note this function does NOT acquire the lock.
        """
        with self.__locks_lock:
            #print "Locking %d" % self.id.id
            name = h5i.get_name(self.id)
            if name is None:
                key = self.id.id
            else:
                info = h5g.get_objinfo(self.id)
                key = (info.fileno, info.objno)

            return self.__locks.setdefault(key, threading.RLock())

    lock = property(_get_lock,
        doc = "A threading.RLock instance associated with this HDF5 structure")

class HLObject(LockableObject):

    """
        Base class for high-level interface objects.

        All objects of this class support the following properties:

        id:     Low-level identifer, compatible with the h5py.h5* modules.
        name:   (Some) name of this object in the HDF5 file.
        attrs:  HDF5 attributes of this object.  See the AttributeManager docs.
    """

    name = property(lambda self: h5i.get_name(self.id),
        doc = "Name of this object in the HDF5 file.  Not necessarily unique.")
    attrs = property(lambda self: self._attrs,
        doc = "Provides access to HDF5 attributes. See AttributeManager.")

    def __repr__(self):
        return str(self)

    def __nonzero__(self):
        return self.id.__nonzero__()

class Group(HLObject):

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

    names = property(lambda self: tuple(self),
        doc = "Tuple of group member names")

    def __init__(self, parent_object, name, create=False):
        """ Create a new Group object, from a parent object and a name.

            If "create" is False (default), try to open the given group,
            raising an exception if it doesn't exist.  If "create" is True,
            create a new HDF5 group and link it into the parent group.
        """
        with parent_object.lock:
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

            This limitation is intentional, and may be lifted in the future.
        """
        with self.lock:
            if isinstance(obj, Group) or isinstance(obj, Dataset) or isinstance(obj, Datatype):
                self.id.link(h5i.get_name(obj.id), name, link_type=h5g.LINK_HARD)

            elif isinstance(obj, numpy.dtype):
                htype = h5t.py_create(obj)
                htype.commit(self.id, name)

            else:
                if not isinstance(obj, numpy.ndarray):
                    obj = numpy.array(obj)
                Dataset(self, name, data=obj)

    def __getitem__(self, name):
        """ Open an object attached to this group. 

            Currently can open groups, datasets, and named types.
        """
        with self.lock:
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

    def iteritems(self):
        """ Iterate over the group members as (name, value) pairs """
        with self.lock:
            for name in self:
                yield (name, self[name])

    def create_group(self, name):
        """ Create and return a subgroup.

            Fails if the group already exists.
        """
        return Group(self, name, create=True)

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
        """
        return Dataset(self, name, *args, **kwds)


    def desc(self):
        """ Extended (multi-line) description of this group, as a string.
        """
        with self.lock:
            outstr = 'Group "%s" in file "%s":' % \
                    (hbasename(h5i.get_name(self.id)), os.path.basename(h5f.get_name(self.id)))
            outstr = strhdr(outstr)
            infodct = {"Members": len(self)}
            grpinfo = h5g.get_objinfo(self.id, '.')
            infodct["mtime"] = grpinfo.mtime
            outstr += strlist([(name, infodct[name]) for name in ("Members", "mtime")])
            
            cmnt = self.id.get_comment('.')
            if cmnt != '':
                outstr += '\nComment:\n'+cmnt
            return outstr
        
    def __str__(self):
        with self.lock:
            try:
                return 'Group "%s" (%d members)' % (hbasename(self.name), len(self))
            except:
                return "Invalid group"


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
        in a "with" block:

            with File(...) as f:
                ... do stuff with f...
            # end block
       
        The file will be closed at the end of the block, regardless of any
        exceptions raised. 
    """

    name = property(lambda self: self._name,
        doc = "File name on disk")
    mode = property(lambda self: self._mode,
        doc = "Python mode used to open file")

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
        self._path = None
        self._rlhist = []  # for readline nonsense

    def close(self):
        """ Close this HDF5 file.  All open objects will be invalidated.
        """
        with self.lock:
            self.id._close()
            self.fid.close()

    def flush(self):
        """ Tell the HDF5 library to flush its buffers.
        """
        h5f.flush(self.fid)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        with self.lock:
            if self.id._valid:
                self.close()
            
    def __str__(self):
        with self.lock:
            try:
                return 'File "%s", root members: %s' % (self.name, ', '.join(['"%s"' % name for name in self]))
            except:
                return "Invalid file"

    def browse(self, dict=None):
        """ Open a command line shell to browse this file. If dict is not
            specified, any imported objects will be placed in the caller's
            global() dictionary.
        """
        if not self.id._valid:
            print "Can't browse: this file is closed."
            return

        if dict is None:
            dict = inspect.currentframe().f_back.f_globals

        def gethist():
            rlhist = [readline.get_history_item(x) for x in xrange(readline.get_current_history_length()+1)]
            rlhist = [x for x in rlhist if x is not None]
            return rlhist

        # The following is an ugly hack to prevent readline from mixing the cmd
        # session with IPython's session.  Is there a better way to do this?
        if readline:
            hist = gethist()
            readline.clear_history()
            for x in self._rlhist:
                readline.add_history(x)
        try:
            browser = _H5Browser(self, self._path, importdict=dict)
        finally:
            if readline:
                self._rlhist.extend(gethist())
                readline.clear_history()
                for x in hist:
                    readline.add_history(x)
        self._path = browser.path

class Dataset(HLObject):

    """ High-level interface to an HDF5 dataset.

        Dataset(group, name, data=None, dtype=None, shape=None, **kwds)

        Datasets behave superficially like Numpy arrays.  The full Numpy
        slicing syntax, including recarray indexing of named fields (even
        more than one), is supported.  The object returned is always a
        Numpy ndarray.

        Additionally, the following properties are provided:
          shape:    Numpy-style shape tuple of dimsensions
          dtype:    Numpy dtype representing the datatype
          value:    Copy of the full dataset, as either a Numpy array or a
                     Numpy/Python scalar, depending on the shape.
    """

    shape = property(lambda self: self.id.shape,
        doc = "Numpy-style shape tuple giving dataset dimensions")

    dtype = property(lambda self: self.id.dtype,
        doc = "Numpy dtype representing the datatype")

    def _getval(self):
        with self.lock:
            arr = self[...]
            if arr.shape == ():
                return numpy.asscalar(arr)
            return arr

    value = property(_getval,
        doc = "The entire dataset, as an array or scalar depending on the shape.")

    def __init__(self, group, name,
                    shape=None, dtype=None, data=None,
                    chunks=None, compression=None, shuffle=False, fletcher32=False):
        """ Construct a Dataset object.  You might find it easier to use the
            Group methods: Group["name"] or Group.create_dataset().

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

            Creation keywords (* is default); "chunks" is required for all:

            chunks:        Tuple of chunk dimensions or None*
            compression:   DEFLATE (gzip) compression level, int or None*
            shuffle:       Use the shuffle filter? (requires compression) T/F*
            fletcher32:    Enable Fletcher32 error detection? T/F*
        """
        with group.lock:
            if data is None and shape is None:
                if any((data,dtype,shape,chunks,compression,shuffle,fletcher32)):
                    raise ValueError('You cannot specify keywords when opening a dataset.')
                self.id = h5d.open(group.id, name)
            else:
                if ((data is None) and (shape is None)) or \
                   ((data is not None) and (shape is not None)):
                    raise ValueError("*Either* data *or* the shape must be specified.")
                
                if data is not None:
                    shape = data.shape
                    dtype = data.dtype
                else:
                    if dtype is None:
                        dtype = "=f4"
                
                dtype = numpy.dtype(dtype)

                plist = h5p.create(h5p.DATASET_CREATE)
                if chunks:
                    plist.set_chunk(chunks)
                if shuffle:
                    plist.set_shuffle()
                if compression is not None:
                    if compression is True:  # prevent accidental abuse
                        compression = 6
                    plist.set_deflate(compression)
                if fletcher32:
                    plist.set_fletcher32()

                space_id = h5s.create_simple(shape)
                type_id = h5t.py_create(dtype)

                self.id = h5d.create(group.id, name, type_id, space_id, plist)
                if data is not None:
                    self.id.write(h5s.ALL, h5s.ALL, data)

            self._attrs = AttributeManager(self)

    def __getitem__(self, args):
        """ Read a slice from the HDF5 dataset.  Takes slices and
            recarray-style field names (more than one is allowed!) in any
            order.

            For a compound dataset ds, with shape (10,10,5) and fields "a", "b" 
            and "c", the following are all legal syntax:

            ds[1,2,3]
            ds[1,2,:]
            ds[...,3]
            ds[1]
            ds[:]
            ds[1,2,3,"a"]
            ds[0:5:2, ..., 0:2, "a", "b"]
        """
        with self.lock:
            start, count, stride, names = slicer(self.shape, args)

            if not (len(start) == len(count) == len(stride) == self.id.rank):
                raise ValueError("Indices do not match dataset rank (%d)" % self.id.rank)

            htype = self.id.get_type()
            if len(names) > 0:
                if htype.get_class() == h5t.COMPOUND:

                    subtypes = {}
                    for idx in range(htype.get_nmembers()):
                        subtypes[htype.get_member_name(idx)] = htype.get_member_type(idx)

                    for name in names:
                        if name not in subtypes:
                            raise ValueError("Field %s does not appear in this type." % name)

                    insertlist = [(name, subtypes[name].get_size()) for name in names]
                    totalsize = sum([x[1] for x in insertlist])

                    mtype = h5t.create(h5t.COMPOUND, totalsize)

                    offset = 0
                    for name, size in insertlist:
                        mtype.insert(name, offset, subtypes[name])
                        offset += size
                else:
                    raise ValueError("This dataset has no named fields.")
            else:
                mtype = htype

            fspace = self.id.get_space()
            if fspace.get_simple_extent_type() == h5s.SCALAR:
                fspace.select_all()
            else:
                fspace.select_hyperslab(start, count, stride)
            mspace = h5s.create_simple(count)

            arr = numpy.ndarray(count, mtype.dtype)

            self.id.read(mspace, fspace, arr)

            if len(names) == 1:
                # Match Numpy convention for recarray indexing
                return arr[names[0]].squeeze()
            return arr.squeeze()

    def __setitem__(self, args, val):
        """ Write to the HDF5 dataset from an Numpy array.  The shape of the
            Numpy array must match the shape of the selection, and the Numpy
            array's datatype must be convertible to the HDF5 datatype.
        """
        with self.lock:
            start, count, stride, names = slicer(self.shape, args)
            if len(names) != 0:
                raise ValueError("Field name selections are not allowed for write.")

            val = numpy.array(val, dtype=self.dtype)

            if count != val.shape:
                # Allow assignments (1,10) => (10,)
                if numpy.product(count) != numpy.product(val.shape):
                    raise ValueError("Selection (%s) must be compatible with target (%s)" % (str(count), str(val.shape)))
                else:
                    val = val.reshape(count)

            fspace = self.id.get_space()
            fspace.select_hyperslab(start, count, stride)
            mspace = h5s.create_simple(val.shape)

            self.id.write(mspace, fspace, numpy.array(val))

    def __str__(self):
        with self.lock:
            try:
                return 'Dataset "%s": %s %s' % (hbasename(self.name),
                        str(self.shape), repr(self.dtype))
            except:
                return "Invalid dataset"

class AttributeManager(LockableObject):

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

    names = property(lambda self: tuple(self),
        doc = "Tuple of attribute names")

    def __init__(self, parent):
        """ Private constructor; you should not create these.
        """
        self.id = parent.id

    def __getitem__(self, name):
        """ Read the value of an attribute.  If the attribute is scalar, it
            will be returned as a Numpy scalar.  Otherwise, it will be returned
            as a Numpy ndarray.
        """
        with self.lock:
            attr = h5a.open_name(self.id, name)

            arr = numpy.ndarray(attr.shape, dtype=attr.dtype)
            attr.read(arr)

            if len(arr.shape) == 0:
                return numpy.asscalar(arr)
            return arr

    def __setitem__(self, name, value):
        """ Set the value of an attribute, overwriting any previous value.
            The value you provide must be convertible to a Numpy array or
            scalar.  If it's not, the action is aborted with no data loss.

            Any existing value is destroyed just before the call to h5a.create.
            If the creation fails, the data is not recoverable.
        """
        with self.lock:
            if not isinstance(value, numpy.ndarray):
                value = numpy.array(value)

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
        return h5a.get_num_attrs(self.id)


    def __iter__(self):
        """ Iterate over the names of attributes. """
        with self.lock:
            for name in h5a.py_listattrs(self.id):
                yield name

    def iteritems(self):
        """ Iterate over (name, value) tuples. """
        with self.lock:
            for name in self:
                yield (name, self[name])

    def __contains__(self, name):
        """ Determine if an attribute exists, by name. """
        return h5a.py_exists(self.id, name)

    def __str__(self):
        with self.lock:
            try:
                rstr = 'Attributes of "%s": ' % hbasename(h5i.get_name(self.id))
                if len(self) == 0:
                    rstr += '(none)'
                else:
                    rstr += ', '.join(['"%s"' % x for x in self])
                return rstr
            except:
                return "Invalid attributes object"

    def __repr__(self):
        return str(self)

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

    dtype = property(lambda self: self.id.dtype,
        doc = "Numpy dtype equivalent for this datatype")

    def __init__(self, grp, name):
        """ Private constructor; you should not create these.
        """
        with grp.lock:
            self.id = h5t.open(grp.id, name)
            self._attrs = AttributeManager(self)

    def __str__(self):
        with self.lock:
            try:
                return "Named datatype object (%s)" % str(self.dtype)
            except:
                return "Invalid datatype object"




