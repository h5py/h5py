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

        A Dataset object is designed to permit "Numpy-like" access to the 
        underlying HDF5 dataset.  It supports array-style indexing, which 
        returns Numpy ndarrays.  "Extended-recarray" slicing is also possible;
        specify the names of fields you want along with the numerical slices.
        The underlying array can also be written to using the indexing syntax.

        HDF5 attribute access is provided through the property obj.attrs.  See
        the AttributeManager class documentation for more information.

        Read-only properties:
        shape       Tuple containing array dimensions
        dtype       A Numpy dtype representing the array data-type.

        Writable properties:
        cnames:     HDF5 compound names used for complex I/O.  This can be
                    None, (), or a 2-tuple with ("realname", "imgname").
    """

    # --- Properties (Dataset) ------------------------------------------------

    #: Numpy-style shape tuple giving dataset dimensions
    shape = property(lambda self: h5d.py_shape(self.id))

    #: Numpy dtype representing the datatype
    dtype = property(lambda self: h5d.py_dtype(self.id))

    def _set_byteorder(self, order):
        if order is not None:
            h5t._validate_byteorder(order)
        self._byteorder = order
    
    #: Set to <, > or = to coerce I/0 to a particular byteorder, or None to use default.
    byteorder = property(lambda self: self._byteorder, _set_byteorder)

    def _set_cnames(self, names):
        if names is not None:
            h5t._validate_complex(names)
        self._cnames = names

    #: Set to (realname, imgname) to control I/O of Python complex numbers.
    cnames = property(lambda self: self._cnames, _set_cnames)

    #: Provides access to HDF5 attributes. See AttributeManager docstring.
    attrs = property(lambda self: self._attrs)

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
            self.id = h5d.py_create(group.id, name, data, shape, 
                                    chunks, compression, shuffle, fletcher32)

        self._attrs = AttributeManager(self)
        self._byteorder = None
        self._cnames = None

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

        if names is not None and self.dtype.names is None:
            raise ValueError('This dataset has no named fields.')
        tid = 0
        try:
            tid = h5d.get_type(self.id)
            dt = h5t.py_translate_h5t(tid, byteorder=self._byteorder,
                                     compound_names=names,
                                     complex_names=self._cnames)
        finally:
            if tid != 0:
                h5t.close(tid)

        arr = h5d.py_read_slab(self.id, start, count, stride, dtype=dt)
        if names is not None and len(names) == 1:
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
        if names is not None:
            raise ValueError("Field names are not allowed for write.")

        h5d.py_write_slab(self.id, args[-1], start, stride)

    def close(self):
        """ Force the HDF5 library to close and free this object. This
            will be called automatically when the object is garbage collected,
            if it hasn't already.
        """
        h5d.close(self.id)

    def __del__(self):
        if h5i.get_type(self.id) == h5i.DATASET:
            h5d.close(self.id)

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


# === Browsing and interactivity ==============================================

import inspect
import string
import posixpath


class _H5Browse(object):

    def __init__(self):
        self.filename = None
        self.file_obj = None
        self.path = None

    def _loadfile(self, filename):
        if self.file_obj is not None:
            self.file_obj.close()
            self.filename = None

        self.file_obj = File(filename, 'r+')
        self.filename = filename

    def __call__(self, filename=None, importdict=None):
        """ Browse a new file, or the current one.
        """
        if filename is not None:
            self._loadfile(filename)
        else:
            if self.file_obj is None:
                raise ValueError("Must provide filename if no file is currently open")

        if importdict is None:  # hang on tight... here we go...
            importdict = inspect.currentframe().f_back.f_globals

        cmdinstance = _H5Cmd(self.file_obj, self.filename, importdict, self.path)
        cmdinstance.browse()
        self.path = cmdinstance.path

class _H5Cmd(cmd.Cmd):

    def __init__(self, file_obj, filename, importdict, groupname=None):
        cmd.Cmd.__init__(self)
        self.file = file_obj
        self.filename = filename

        if groupname is None:
            groupname = '/'
        self.group = self.file[groupname]
        self.path = groupname

        self.prompt = os.path.basename(self.filename)+' '+os.path.basename(self.path)+'> '

        self.importdict = importdict

    def browse(self):
        self.cmdloop('Browsing "%s". Type "help" for commands, "exit" to exit.' % os.path.basename(self.filename))

    def _safename(self, name):
        legal = string.ascii_letters + '0123456789'
        instring = list(name)
        for idx, x in enumerate(instring):
            if x not in legal:
                instring[idx] = '_'
        if instring[0] not in string.ascii_letters:
            instring = ['_']+instring
        return ''.join(instring)

    def do_ls(self, line):

        def padline(line, width, trunc=True):
            slen = len(line)
            if slen >= width:
                if trunc:
                    line = line[0:width-4]+'... '
                else:
                    line = line+' '
            else:
                line = line + ' '*(width-slen)
            return line

        extended = False
        trunc = True
        if line.strip() == '-l':
            extended = True
        if line.strip() == '-ll':
            extended = True
            trunc = False

        for name in self.group:
            outstring = name
            type_code = h5g.get_objinfo(self.group.id, name).type
            if type_code == h5g.GROUP:
                outstring += "/"

            if extended:
                outstring = padline(outstring, 20, trunc)
                codestring = str(self.group[name])
                outstring += padline(codestring, 60, trunc)

            print outstring

    def do_cd(self, path):
        """ cd <path>
        """
        path = posixpath.normpath(posixpath.join(self.path, path))
        try:
            group = Group(self.file, path)
            self.prompt = os.path.basename(self.filename)+' '+os.path.basename(path)+'> '
        except H5Error, e:
            print e.message
        self.path = path
        self.group = group

    def do_import(self, line):
        if self.importdict is None:
            print "Can't import variables (no import dict provided)."
        line = line.strip()
        objname, as_string, newname = line.partition(' as ')
        newname = newname.strip()
        objname = objname.strip()
        if len(newname) == 0:
            newname = objname
        try:
            self.importdict[newname] = self.group[objname]
        except H5Error, e:
            print e.message

    def do_exit(self, line):
        return True

    def do_EOF(self, line):
        return self.do_exit(line)

    def do_pwd(self, line):
        print self.path

    def complete_import(self, text, line, begidx, endidx):
        return [x for x in self.group if x.find(text)==0]

    def complete_cd(self, text, line, begidx, endidx):
        return [x for x in self.group if x.find(text)==0 \
                    and h5g.get_objinfo(self.group.id,x).type == h5g.GROUP]

    def help_cd(self):
        print ""
        print "cd <name>"
        print "    Enter a subgroup of the current group"
        print ""

    def help_pwd(self):
        print ""
        print "pwd"
        print "    Print current path"
        print ""

    def help_ls(self):
        print ""
        print "ls [-l] [-ll]"
        print "    Print the contents of the current group."
        print "    Optional long format with -l (80 columns)"
        print "    Very long format (-ll) has no column limit."
        print ""

    def help_import(self):
        print ""
        print "import <name> [as <python_name>]"
        print "    Import a member of the current group as a Python object" 
        print "    at the interactive level, optionally under a different"
        print "    name."
        print ""



# === Utility functions =======================================================

def _open_arbitrary(group_obj, name):
    """ Figure out the type of an object attached to an HDF5 group and return 
        the appropriate high-level interface object.

        Currently supports Group, Dataset, and NamedDatatype
    """
    info = h5g.get_objinfo(group_obj.id, name)

    if info.type == h5g.GROUP:      # group
        return Group(group_obj, name)

    elif info.type == h5g.DATASET:  # dataset
        return Dataset(group_obj, name)

    elif info.type == h5g.DATATYPE: # named type
        return NamedDatatype(group_obj, name)

    raise NotImplementedError('Object type "%s" unsupported by the high-level interface.' % h5g.PY_TYPE[info.type])

def slicer(shape, args):
    """ Processes arguments to __getitem__ methods.  
    
        shape:  Dataset shape (tuple)
        args:   Raw __getitem__ args; integers, slices or strings in any order.
        
        Returns 4-tuple:
        (start, count, stride, names)
        Start/count/stride are guaranteed not to be None.
        Names will either be None or a list of non-zero length.
    """

    if not isinstance(args, tuple):
        args = (args,)

    rank = len(shape)
    
    slices = []     # Holds both slice objects and integer indices.
    names = []      # Field names (strings)

    # Sort slice-like arguments from strings
    for arg in args:
        if isinstance(arg, int) or isinstance(arg, long) or isinstance(arg, slice):
            slices.append(arg)
        elif isinstance(arg, str):
            names.append(arg)
        else:
            raise TypeError("Unsupported slice type (must be int/long/slice/str): %s" % repr(arg))

    # If there are no names, this is interpreted to mean "all names."  So
    # return None instead of an empty sequence.
    if len(names) == 0:
        names = None
    else:
        names = tuple(names)

    # Check for special cases

    # 1. No numeric slices == full dataspace
    if len(slices) == 0:
            return ((0,)*rank, shape, (1,)*rank, names)

    # 2. Single numeric slice ":" == full dataspace
    if len(slices) == 1 and isinstance(slices[0], slice):
        slice_ = slices[0]
        if slice_.stop == None and slice_.step == None and slice_.stop == None:
            return ((0,)*rank, shape, (1,)*rank, names)

    # Validate slices
    if len(slices) != rank:
        raise ValueError("Number of numeric slices must match dataset rank (%d)" % rank)

    start = []
    count = []
    stride = []

    # Parse slices to assemble hyperslab start/count/stride tuples
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

    return (tuple(start), tuple(count), tuple(stride), names)


#: Command-line HDF5 file "shell": browse(name) (or browse() for last file).
browse = _H5Browse()













