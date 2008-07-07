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

import numpy
import inspect

from h5py import h5, h5f, h5g, h5s, h5t, h5d, h5a, h5p, h5z, h5i
from h5py.h5 import H5Error
from utils_hl import slicer, hbasename
from browse import _H5Browser

try:
    # For interactive File.browse() capability
    import readline
except ImportError:
    readline = None

class HLObject(object):

    name = property(lambda self: h5i.get_name(self.id),
        doc = "Name of this object in the HDF5 file.  Not necessarily unique.")

    def __repr__(self):
        return str(self)

class Group(HLObject):

    """ Represents an HDF5 group.

        Iterating over a group yields the names of its members.
    """

    attrs = property(lambda self: self._attrs,
        doc = "Provides access to HDF5 attributes. See AttributeManager.")

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
        """
        if isinstance(obj, Group) or isinstance(obj, Dataset) or isinstance(obj, Datatype):
            self.id.link(name, h5i.get_name(obj.id), link_type=h5g.LINK_HARD)

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
        info = self.id.get_objinfo(name)

        if info.type == h5g.DATASET:
            dset = Dataset(self, name)
            return dset

        elif info.type == h5g.GROUP:
            return Group(self, name)

        elif info.type == h5g.TYPE:
            return Datatype(self, name)

        raise ValueError("Don't know how to open object of type %d" % info.type)

    def __delitem__(self, name):
        """ Delete (unlink) an item from this group. """
        self.id.unlink(name)

    def __len__(self):
        return self.id.get_num_objs()

    def __iter__(self):
        return self.id.py_iter()

    def __str__(self):
        if self.id._valid:
            return 'Group "%s" (%d members): %s' % (hbasename(self.name),
                    len(self), ', '.join(['"%s"' % name for name in self]))
        return "Closed group"

    def iteritems(self):
        for name in self:
            yield (name, self[name])

class File(Group):

    """ Represents an HDF5 file on disk.

        Created with standard Python syntax File(name, mode).
        Legal modes: r, r+, w, w+, a  (default 'r')

        File objects inherit from Group objects; Group-like methods all
        operate on the HDF5 root group ('/').  Like Python file objects, you
        must close the file ("obj.close()") when you're done with it.
    """

    name = property(lambda self: self._name,
        doc = "File name on disk")
    mode = property(lambda self: self._mode,
        doc = "Python mode used to open file")

    _modes = ('r','r+','w','w+','a')

    # --- Public interface (File) ---------------------------------------------

    def __init__(self, name, mode='r', noclobber=False):
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
        plist.set_fclose_degree(h5f.CLOSE_STRONG)
        if mode == 'r':
            self.fid = h5f.open(name, h5f.ACC_RDONLY, accesslist=plist)
        elif 'r' in mode or 'a' in mode:
            self.fid = h5f.open(name, h5f.ACC_RDWR, accesslist=plist)
        elif noclobber:
            self.fid = h5f.create(name, h5f.ACC_EXCL, accesslist=plist)
        else:
            self.fid = h5f.create(name, h5f.ACC_TRUNC, accesslist=plist)

        self.id = self.fid  # So the Group constructor can find it.
        Group.__init__(self, self, '/')
    
        self._name = name
        self._mode = mode
        self._path = None
        self._rlhist = []  # for readline nonsense

    def close(self):
        """ Close this HDF5 file.  All open identifiers will become invalid.
        """
        self.id.close()
        self.fid.close()

    def flush(self):
        """ Tell the HDF5 library to flush its buffers.
        """
        h5f.flush(self.fid)

    def __str__(self):
        if self.id._valid:
            return 'File "%s", root members: %s' % (self.name, ', '.join(['"%s"' % name for name in self]))
        return "Closed file (%s)" % self.name

    def browse(self, dict=None):
        """ Browse this file. If no import dict is provided, will seize the
            caller's global dictionary.
        """
        if dict is None:
            dict = inspect.currentframe().f_back.f_globals

        def gethist():
            rlhist = [readline.get_history_item(x) for x in xrange(readline.get_current_history_length())]
            rlhist = [x for x in rlhist if x is not None]
            return rlhist

        # The following is an ugly hack to prevent readline from mixing the cmd
        # session with IPython's session.  Is there a better way to do this?
        hist = None
        if readline:
            hist = gethist()
            readline.clear_history()
            for x in self._rlhist:
                readline.add_history(x)
        try:
            browser = _H5Browser(self, self._path, importdict=dict)
        finally:
            if hist:
                self._rlhist.extend(gethist())
                readline.clear_history()
                for x in hist:
                    readline.add_history(x)
        self._path = browser.path

class Dataset(HLObject):

    """ High-level interface to an HDF5 dataset
    """

    shape = property(lambda self: self.id.shape,
        doc = "Numpy-style shape tuple giving dataset dimensions")

    dtype = property(lambda self: self.id.dtype,
        doc = "Numpy dtype representing the datatype")

    attrs = property(lambda self: self._attrs,
        doc = "Provides access to HDF5 attributes. See AttributeManager.")

    def _getval(self):
        arr = self[...]
        if arr.shape == ():
            return numpy.asscalar(arr)
        return arr

    value = property(_getval,
        doc = "The entire dataset, as an array or scalar depending on the shape.")

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
            order.

            For a compound dataset ds, with shape (10,10,5) and fields "a", "b" 
            and "c", the following are all legal subscripts:

            ds[1,2,3]
            ds[1,2,:]
            ds[...,3]
            ds[1]
            ds[:]
            ds[1,2,3,"a"]
            ds[0:5:2, 0:6:3, 0:2, "a", "b"]
        """
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
            return arr[names[0]]
        return arr

    def __setitem__(self, args):
        """ Write to the underlying array from an existing Numpy array.  The
            shape of the Numpy array must match the shape of the selection,
            and the Numpy array's datatype must be convertible to the HDF5
            array's datatype.
        """
        val = args[-1]
        args = args[0:-1]

        start, count, stride, names = slicer(self.shape, args)
        if len(names) != 0:
            raise ValueError("Field name selections are not allowed for write.")

        if count != val.shape:
            raise ValueError("Selection shape (%s) must match target shape (%s)" % (str(count), str(val.shape)))

        fspace = self.id.get_space()
        fspace.select_hyperslab(start, count, stride)
        mspace = h5s.create_simple(val.shape)

        self.id.write(mspace, fspace, array(val))

    def __str__(self):
        if self.id._valid:
            return 'Dataset "%s": %s %s' % (hbasename(self.name),
                    str(self.shape), repr(self.dtype))
        return "Closed dataset"

class AttributeManager(HLObject):

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
    def __init__(self, parent):

        self.id = parent.id

    def __getitem__(self, name):
        attr = h5a.open_name(self.id, name)

        arr = numpy.ndarray(attr.shape, dtype=attr.dtype)
        attr.read(arr)

        if len(arr.shape) == 0:
            return numpy.asscalar(arr)
        return arr

    def __setitem__(self, name, value):
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value)

        space = h5s.create_simple(value.shape)
        htype = h5t.py_create(value.dtype)

        # TODO: some kind of transaction safeguard here
        try:
            h5a.delete(self.id, name)
        except H5Error:
            pass
        attr = h5a.create(self.id, name, htype, space)
        attr.write(value)

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
        if self.id._valid:
            rstr = 'Attributes of "%s": ' % hbasename(self.name)
            if len(self) == 0:
                rstr += '(none)'
            else:
                rstr += ', '.join(['"%s"' % x for x in self])
        else:
            rstr = "Attributes of closed object."

        return rstr

    def __repr__(self):
        return str(self)

class Datatype(HLObject):

    """
        Represents an HDF5 datatype.

        These intentionally only represent named types.
    """

    dtype = property(lambda self: self.id.dtype)

    def __init__(grp, name):
        self.id = h5t.open(grp.id, name)

    def __str__(self):
        if self.id._valid:
            return "Named datatype object (%s)" % str(self.dtype)
        return "Closed datatype object"




