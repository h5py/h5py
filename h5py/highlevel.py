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

    Highlights:

    - Groups provide dictionary-like __getitem__ access to their members, and 
      allow iteration over member names.  File objects also perform these
      operations, implicitly on the root ('/') group.

    - Datasets support Numpy dtype and shape access, along with read/write 
      access to the underlying HDF5 dataset (including slicing for partial I/0),
      and reading/writing specified fields of a compound type object.

    - Both group and dataset attributes can be accessed via a dictionary-style
      attribute manager (group_obj.attrs and dataset_obj.attrs).  See the
      highlevel.AttributeManager docstring for more information.

    - Named datatypes are reached through the NamedType class, which allows
      attribute access like Group and Dataset objects.

    - An interactive command-line utility "browse(filename)" allows access to 
      HDF5 datasets, with commands like "cd, ls", etc.  Also allows you to
      import groups and datasets directly into Python as highlevel.Group and
      highlevel.Dataset objects.

    - There are no "h5py.highlevel" exceptions; classes defined here raise 
      native Python exceptions.  However, they will happily propagate 
      exceptions from the underlying h5py.h5* modules, which are (mostly) 
      subclasses of h5py.errors.H5Error.
"""

__revision__ = "$Id$"

import os
import cmd
import random
import string
import numpy

import h5f
import h5g
import h5i
import h5d
import h5t
import h5a
import h5p
from h5e import H5Error

# === Main classes (Dataset/Group/File) =======================================

class Dataset(object):

    """ High-level interface to an HDF5 dataset
    """

    # --- Properties (Dataset) ------------------------------------------------

    shape = property(lambda self: self.id.shape,
        doc = "Numpy-style shape tuple giving dataset dimensions")

    dtype = property(lambda self: self.id.dtype,
        doc = "Numpy dtype representing the datatype")

    attrs = property(lambda self: self._attrs,
        doc = "Provides access to HDF5 attributes")

    # --- Public interface (Dataset) ------------------------------------------

    def __init__(self, group, name,
                    data=None, dtype=None, shape=None, 
                    chunks=None, compression=None, shuffle=False, fletcher32=False):
        """ Create a new Dataset object.  There are two modes of operation:

            1.  Open an existing dataset
                If you only supply the required parameters "group" and "name",
                the object will attempt to open an existing HDF5 dataset.

            2.  Create a dataset
                You can supply either:
                - Keyword "data"; a Numpy array from which the shape, dtype and
                    initial contents will be determined.
                - Both "dtype" (Numpy dtype object) and "shape" (tuple of 
                    dimensions).

            Creating a dataset will fail if another of the same name already 
            exists.  Also, chunks/compression/shuffle/fletcher32 may only be
            specified when creating a dataset.

            Creation keywords (* is default):

            chunks:        Tuple of chunk dimensions or None*
            compression:   DEFLATE (gzip) compression level, int or None*
            shuffle:       Use the shuffle filter? (requires compression) T/F*
            fletcher32:    Enable Fletcher32 error detection? T/F*
        """
        if data is None and dtype is None and shape is None:
            if any((data,dtype,shape,chunks,compression,shuffle,fletcher32)):
                raise ValueError('You cannot specify keywords when opening a dataset.')
            self.id = h5d.open(group.id, name)
        else:
            if ((data is None) and not (shape and dtype)) or \
               ((data is not None) and (shape or dtype)):
                raise ValueError("Either data or both shape and dtype must be specified.")
            
            if data is not None:
                shape = data.shape
                dtype = data.dtype

            plist = h5p.create(h5p.DATASET_CREATE)
            if chunks:
                plist.set_chunks(chunks)
            if shuffle:
                plist.set_shuffle()
            if compression:
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
        """ Read a slice from the underlying HDF5 array.  Takes slices and
            recarray-style field names (more than one is allowed!) in any
            order.  Examples:

            ds[0,0:15,:] => (1 x 14 x <all) slice on 3-dimensional dataset.

            ds[:] => All elements, regardless of dimension.

            ds[0:3, 1:4, "a", "b"] => (3 x 3) slice, only including compound
                                      elements "a" and "b", in that order.
        """
        start, count, stride, names = slicer(self.shape, args)

        if not (len(start) == len(count) == len(stride) == self.id.rank):
            raise ValueError("Indices do not match dataset rank (%d)" % self.id.rank)

        htype = self.id.get_type()
        if len(names) > 0:
            if htype.get_class() == h5t.COMPOUND:
                mtype = h5t.create(h5t.COMPOUND)

                offset = 0
                for idx in range(htype.get_nmembers()):
                    hname = htype.get_member_name(idx)
                    if hname in names:
                        subtype = h5type.get_member_type(idx)
                        mtype.insert(hname, offset, subtype)
                        offset += subtype.get_size()
            else:
                raise ValueError("This dataset has no named fields.")
        else:
            mtype = htype

        fspace = self.id.get_space()
        fspace.select_hyperslab(start, count, stride)
        mspace = h5s.create_simple(count)

        arr = ndarray(count, mtype.dtype)

        self.id.read(mspace, fspace, arr)

        if len(names) == 1
            # Match Numpy convention for recarray indexing
            return arr[names[0]]
        return arr

    def __setitem__(self, args):
        """ Write to the underlying array from an existing Numpy array.  The
            shape of the Numpy array must match the shape of the selection,
            and the Numpy array's datatype must be convertible to the HDF5
            array's datatype.
        """
        val = args[-1]
        start, count, stride, names = slicer(val.shape, args[:-1])
        if len(names) > 0:
            raise ValueError("Field names are not allowed for write.")

        self.id.
        h5d.py_write_slab(self.id, args[-1], start, stride)


    def __str__(self):
        return 'Dataset: '+str(self.shape)+'  '+repr(self.dtype)

    def __repr__(self):
        return self.__str__()

class Group(object):
    """ Represents an HDF5 group object

        Group members are accessed through dictionary-style syntax.  Iterating
        over a group yields the names of its members, while the method
        iteritems() yields (name, value) pairs. Examples:

            highlevel_obj = group_obj["member_name"]
            member_list = list(group_obj)
            member_dict = dict(group_obj.iteritems())

        - Accessing items: generally a Group or Dataset object is returned. In
          the special case of a scalar dataset, a Numpy array scalar is
          returned.

        - Setting items:
            1. Existing Group or Dataset: create a hard link in this group
            2. Numpy array: create a new dataset here, overwriting any old one
            3. Anything else: try to create a Numpy array.  Also works with
               Python scalars which have Numpy type equivalents.

        - Deleting items: unlinks the object from this group.

        - Attribute access: through the property obj.attrs.  See the 
          AttributeManager class documentation for more information.

        - len(obj) returns the number of group members
    """

    #: Provides access to HDF5 attributes. See AttributeManager docstring.
    attrs = property(lambda self: self._attrs)

    # --- Public interface (Group) --------------------------------------------

    def __init__(self, parent_object, name, create=False):
        """ Create a new Group object, from a parent object and a name.

            If "create" is False (default), try to open the given group,
            raising an exception if it doesn't exist.  If "create" is True,
            create a new HDF5 group and link it into the parent group.
        """
        if create:
            self.id = h5g.create(parent_object.id, name)
        else:
            self.id = h5g.open(parent_object.id, name)
        
        #: Group attribute access (dictionary-style)
        self._attrs = AttributeManager(self)

    def __delitem__(self, name):
        """ Unlink a member from the HDF5 group.
        """
        h5g.unlink(self.id, name)

    def __setitem__(self, name, obj):
        """ Add the given object to the group.  Here are the rules:

            1. If "obj" is a Dataset or Group object, a hard link is created
                in this group which points to the given object.
            2. If "obj" is a Numpy ndarray, it is converted to a dataset
                object, with default settings (contiguous storage, etc.).
            3. If "obj" is anything else, attempt to convert it to an ndarray
                and store it.  Scalar values are stored as scalar datasets.
                Raise ValueError if we can't understand the resulting array 
                dtype.
        """
        if isinstance(obj, Group) or isinstance(obj, Dataset):
            h5g.link(self.id, name, h5i.get_name(obj.id), link_type=h5g.LINK_HARD)

        else:
            if not isinstance(obj, numpy.ndarray):
                obj = numpy.array(obj)
            if h5t.py_can_convert_dtype(obj.dtype):
                dset = Dataset(self, name, data=obj)
                dset.close()
            else:
                raise ValueError("Don't know how to store data of this type in a dataset: " + repr(obj.dtype))


    def __getitem__(self, name):
        """ Retrive the Group or Dataset object.  If the Dataset is scalar,
            returns its value instead.
        """
        retval = _open_arbitrary(self, name)
        if isinstance(retval, Dataset) and retval.shape == ():
            value = h5d.py_read_slab(retval.id, (), ())
            value = value.astype(value.dtype.type)
            retval.close()
            return value
        return retval

    def __iter__(self):
        """ Yield the names of group members.
        """
        return h5g.py_iternames(self.id)

    def iteritems(self):
        """ Yield 2-tuples of (member_name, member_value).
        """
        for name in self:
            yield (name, self[name])

    def __len__(self):
        return h5g.get_num_objs(self.id)

    def close(self):
        """ Immediately close the underlying HDF5 object.  Further operations
            on this Group object will raise an exception.  You don't typically
            have to use this, as these objects are automatically closed when
            their Python equivalents are deallocated.
        """
        h5g.close(self.id)

    def __del__(self):
        if h5i.get_type(self.id) == h5i.GROUP:
            h5g.close(self.id)

    def __str__(self):
        return 'Group (%d members): ' % self.nmembers + ', '.join(['"%s"' % name for name in self])

    def __repr__(self):
        return self.__str__()

class File(Group):

    """ Represents an HDF5 file on disk.

        Created with standard Python syntax File(name, mode), where mode may be
        one of r, r+, w, w+, a.

        File objects inherit from Group objects; Group-like methods all
        operate on the HDF5 root group ('/').  Like Python file objects, you
        must close the file ("obj.close()") when you're done with it.
    """

    _modes = ('r','r+','w','w+','a')

    # --- Public interface (File) ---------------------------------------------

    def __init__(self, name, mode, noclobber=False):
        """ Create a new file object.  

            Valid modes (like Python's file() modes) are: 
            - 'r'   Readonly, file must exist
            - 'r+'  Read/write, file must exist
            - 'w'   Write, create/truncate file
            - 'w+'  Read/write, create/truncate file
            - 'a'   Read/write, file must exist (='r+')

            If "noclobber" is specified, file truncation (w/w+) will fail if 
            the file already exists.  Note this is NOT the default.
        """
        if not mode in self._modes:
            raise ValueError("Invalid mode; must be one of %s" % ', '.join(self._modes))
              
        plist = h5p.create(h5p.FILE_ACCESS)
        try:
            h5p.set_fclose_degree(plist, h5f.CLOSE_STRONG)
            if mode == 'r':
                self.fid = h5f.open(name, h5f.ACC_RDONLY, access_id=plist)
            elif 'r' in mode or 'a' in mode:
                self.fid = h5f.open(name, h5f.ACC_RDWR, access_id=plist)
            elif noclobber:
                self.fid = h5f.create(name, h5f.ACC_EXCL, access_id=plist)
            else:
                self.fid = h5f.create(name, h5f.ACC_TRUNC, access_id=plist)
        finally:
            h5p.close(plist)

        self.id = self.fid  # So the Group constructor can find it.
        Group.__init__(self, self, '/')

        # For __str__ and __repr__
        self.filename = name
        self.mode = mode
        self.noclobber = noclobber

    def close(self):
        """ Close this HDF5 object.  Note that any further access to objects
            defined in this file will raise an exception.
        """
        if h5i.get_type(self.fid) != h5i.FILE:
            raise IOError("File is already closed.")

        Group.close(self)
        h5f.close(self.fid)

    def flush(self):
        """ Instruct the HDF5 library to flush disk buffers for this file.
        """
        h5f.flush(self.fid)

    def __del__(self):
        """ This docstring is here to remind you that THE HDF5 FILE IS NOT 
            AUTOMATICALLY CLOSED WHEN IT'S GARBAGE COLLECTED.  YOU MUST
            CALL close() WHEN YOU'RE DONE WITH THE FILE.
        """
        pass

    def __str__(self):
        return 'File "%s", root members: %s' % (self.filename, ', '.join(['"%s"' % name for name in self]))

    def __repr_(self):
        return 'File("%s", "%s", noclobber=%s)' % (self.filename, self.mode, str(self.noclobber))


class AttributeManager(object):

    """ Allows dictionary-style access to an HDF5 object's attributes.

        You should never have to create one of these; they come attached to
        Group, Dataset and NamedType objects as "obj.attrs".

        - Access existing attributes with "obj.attrs['attr_name']".  If the
          attribute is scalar, a scalar value is returned, else an ndarray.

        - Set attributes with "obj.attrs['attr_name'] = value".  Note that
          this will overwrite an existing attribute.

        - Delete attributes with "del obj.attrs['attr_name']".

        - Iterating over obj.attrs yields the names of the attributes. The
          method iteritems() yields (name, value) pairs.
        
        - len(obj.attrs) returns the number of attributes.
    """
    def __init__(self, parent_object):
        self.id = parent_object.id

    def __getitem__(self, name):
        obj = h5a.py_get(self.id, name)
        if len(obj.shape) == 0:
            return obj.dtype.type(obj)
        return obj

    def __setitem__(self, name, value):
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value)
        if h5a.py_exists(self.id, name):
            h5a.delete(self.id, name)
        h5a.py_set(self.id, name, value)

    def __delitem__(self, name):
        h5a.delete(self.id, name)

    def __len__(self):
        return h5a.get_num_attrs(self.id)

    def __iter__(self):
        for name in h5a.py_listattrs(self.id):
            yield name

    def iteritems(self):
        for name in self:
            yield (name, self[name])

    def __str__(self):
        return "Attributes: "+', '.join(['"%s"' % x for x in self])

class NamedType(object):

    """ Represents a named datatype, stored in a file.  

        HDF5 datatypes are typically represented by their Numpy dtype
        equivalents; this class exists only to provide access to attributes
        stored on HDF5 named types.  Properties:

        dtype:   Equivalent Numpy dtype for this HDF5 type
        attrs:   AttributeManager instance for attribute access

        Like dtype objects, these are immutable; the worst you can do it
        unlink them from their parent group.
    """ 
        
    def _get_dtype(self):
        if self._dtype is None:
            self._dtype = h5t.py_translate_h5t(self.id)
        return self._dtype

    dtype = property(_get_dtype)

    def __init__(self, group, name, dtype=None):
        """ Open an existing HDF5 named type, or create one.

            If no value is provided for "dtype", try to open a named type
            called "name" under the given group.  If "dtype" is anything
            which can be converted to a Numpy dtype, create a new datatype
            based on it and store it in the group.
        """
        self.id = None
        self._dtype = None  # Defer initialization; even if the named type 
                            # isn't Numpy-compatible, we can still get at the
                            # attributes.

        if dtype is not None:
            dtype = numpy.dtype(dtype)
            tid = h5t.py_translate_dtype(dtype)
            try:
                h5t.commit(group.id, name, tid)
            finally:
                h5t.close(tid)

        self.id = h5t.open(group.id, name)
        self.attrs = AttributeManager(self)

    def close(self):
        """ Force the library to close this object.  It will still exist
            in the file.
        """
        if self.id is not None:
            h5t.close(self.id)
            self.id = None

    def __del__(self):
        if self.id is not None:
            h5t.close(self.id)

# === Utility functions =======================================================

def slicer(shape, args):
    """
        Parse Numpy-style extended slices.  Correctly handle:
        1. Recarray-style field strings (more than one!)
        2. Slice objects
        3. Ellipsis objects
    """
    rank = len(shape)

    if not isinstance(args, tuple):
        args = (args,)
    args = list(args)

    slices = []
    names = []

    # Sort arguments
    for entry in args[:]:
        if isinstance(entry, str):
            names.append(entry)
        else:
            slices.append(entry)

    start = []
    count = []
    stride = []

    # Hack to allow Numpy-style row indexing
    if len(slices) == 1:
        args.append(Ellipsis)

    # Expand integers and ellipsis arguments to slices
    for dim, arg in enumerate(slices):

        if isinstance(arg, int) or isinstance(arg, long):
            if arg < 0:
                raise ValueError("Negative indices are not allowed.")
            start.append(arg)
            count.append(1)
            stride.append(1)

        elif isinstance(arg, slice):

            # slice.indices() method clips, so do it the hard way...

            # Start
            if arg.start is None:
                ss=0
            else:
                if arg.start < 0:
                    raise ValueError("Negative dimensions are not allowed")
                ss=arg.start

            # Stride
            if arg.step is None:
                st = 1
            else:
                if arg.step <= 0:
                    raise ValueError("Only positive step sizes allowed")
                st = arg.step

            # Count
            if arg.stop is None:
                cc = shape[dim]/st
            else:
                if arg.stop < 0:
                    raise ValueError("Negative dimensions are not allowed")
                cc = (arg.stop-ss)/st
            if cc == 0:
                raise ValueError("Zero-length selections are not allowed")

            start.append(ss)
            stride.append(st)
            count.append(cc)

        elif arg == Ellipsis:
            nslices = rank-(len(slices)-1)
            if nslices <= 0:
                continue
            for x in range(nslices):
                idx = dim+x
                start.append(0)
                count.append(shape[dim+x])
                stride.append(1)

        else:
            raise ValueError("Bad slice type %s" % repr(arg))

    return (start, count, stride, names)




