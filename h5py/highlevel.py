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
import posixpath

import h5f
import h5g
import h5i
import h5d
import h5t
import h5a
import h5p
from errors import H5Error

from transactions import Action, TransactionManager, TransactionStateError, \
                         IllegalTransactionError

def tempname(prefix=""):
    return prefix+"".join(random.sample(string.ascii_letters))

# === Base classes / context manager support ==================================

class BaseNamed(object):

    """ Base class for objects which reside in HDF5 files.  Among other things,
        any named object is a valid context manager for Python's "with"
        statement, capable of tracking transactions in HDF5 files.
    """

    def _set_manager(self, val):
        self._manager = val
        if hasattr(self, 'attrs'):
            self.attrs.manager = val
    
    def _get_manager(self):
        return self._manager

    manager = property(_get_manager, _set_manager)

    def __init__(self):
        self.manager = None
        self._tr_active = False  # transaction active?

    def __enter__(self):
        """ Put the object in transaction mode.  If no transaction manager is
            currently associated with the object, create one.

            Please don't call this manually.
        """
        if self._tr_active:
            raise TransactionStateError("A transaction is already in progress.")

        stat = h5g.get_objinfo(self.id, '.')
        token = (stat.fileno, stat.objno)

        if self.manager is None:
            self.manager = TransactionManager()
        
        self.manager.lock(token)
        self._tr_active = True
        return self.manager

    def __exit__(self, type_, value, tb):
        """ Exit transaction mode.  Commits or rolls back, depending on the
            given exception state, but does not destroy the transaction manager.

            Please don't call this manually.
        """
        if not self._tr_active:
            raise TransactionStateError("Exited transaction mode with no transaction in progress")

        if type_ is None:
            self.manager.commit()
        else:
            self.manager.rollback()

        stat = h5g.get_objinfo(self.id, '.')
        token = (stat.fileno, stat.objno)

        self.manager.unlock(token)
        self._tr_active = False

    def begin_transaction(self, manager=None):
        """ Manually put the object into "transaction" mode.  Every API call 
            which can affect the underlying HDF5 state is recorded, and can 
            be reversed.  If the object is already in transaction mode, raises
            TransactionStateError.

            The return value is a TransactionManager instance, which provides 
            methods like commit(), rollback(), etc.
        
            You can optionally use an existing transaction manager.  This lets 
            you track changes across multiple objects, with automatic 
            interlocking to prevent double access to HDF5 entities through 
            multiple Python objects.

            As an alternative to manually beginning and ending transactions,
            you can use any object of this class as the context manager in a 
            Python "with" statement.
        """
        self.manager = manager
        return self.__enter__()

    def end_transaction(self):
        """ Take the object out of "transaction" mode.  Implicitly commits any
            pending actions.  Raises TransactionStateError if the object is not
            in transaction mode.  Never call this inside a "with" block.
        """
        self.__exit__(None, None, None)

class WithWrapper(object):

    """ Create a single context manager out of many named objects, all sharing
        the same transaction manager.
    """

    def __init__(self, *objs):
        self.objs = objs

    def __enter__(self):
        mgr = TransactionManger

        for obj in objs:
            obj.manager = mgr
            obj.__enter__()
            
        return mgr

    def __exit__(self, type_, value, tb):
        return all(obj.__exit__(type_, value, tb) for obj in objs)

def many(*args):
    """ Enables tracking of multiple named objects in Python's "with" statement.
        with_many(obj1, obj2, ...)
    """
    c_mgr = WithWrapper(*args)
    return c_mgr


# === Main classes (Dataset/Group/File) =======================================

class Dataset(BaseNamed):

    """ High-level interface to an HDF5 dataset

        TODO: rework this
    """

    # --- Properties (Dataset) ------------------------------------------------

    def _get_byteorder(self):
        return self._byteorder

    def _set_byteorder(self, val):
        valid = [None, '<', '>', '=']
        if not val in valid:
            raise ValueError("Byte order must be one of %s (got %s)" % (", ".join(valid), str(val)))

        if not self._tr_active:
            self._byteorder = val
        else:
            backup = self._byteorder
            action = Action("Set byte order to " + str(val), 
                    (setattr, (self, '_byteorder', val), {}),
                    (setattr, (self, '_byteorder', backup), ()),
                    None)
            self.manager.do(action)

    def _get_string_length(self):
        return self._string_length

    def _set_string_length(self, val):
        if val is not None and val < 1:
            raise ValueError("String length must be at least 1.")

        if not self._tr_active:
            self._string_length = val
        else:
            backup = self._string_length
            action = Action("Set string length to "+str(val),
                    (setattr, (self, '_string_length', val), {}),
                    (setattr, (self, '_string_length', backup), ()),
                    None)

    def _get_names_mask(self):
        return self._fields

    def _set_names_mask(self, iterable):
        """ Determine which fields of a compound datatype will be read. Only 
            compound fields whose names match those provided by the given 
            iterable will be read.  Any given names which do not exist in the
            HDF5 compound type are simply ignored.

            If the argument is a single string, it will be correctly processed
            (i.e. not exploded).
        """
        if iterable == None:
            val = None
        else:
            if isinstance(iterable, basestring):
                iterable = (iterable,)    # not 'i','t','e','r','a','b','l','e'
            val = tuple(iterable)

        if not self._tr_active:
            self._fields = val
        else:
            backup = self._string_length
            action = Action("Set names mask to \"%s\"" % str(val),
                    (setattr, (self, '_fields', val), {}),
                    (setattr, (self, '_fields', backup), ()),
                    None)


    #: Byte order for data read/written by this object; can be None, <, >, =.
    byteorder = property(_get_byteorder, _set_byteorder)

    #: Convert vlen strings to fixed-width: None or >= 1.
    string_length = property(_get_string_length, _set_string_length)

    #: Restrict I/0 to these fields.  None, iterable of strings, or single string.
    names_mask = property(_get_names_mask, _set_names_mask)

    #: Numpy-style shape tuple for this dataset. Readonly.
    shape = property(lambda self: h5d.py_shape(self.id))

    #: Numpy dtype representing dataset's type. Readonly.
    dtype = property(lambda self: h5d.py_dtype(self.id))

    #: Attribute manager; see AttributeManager docstring.
    attrs = property(lambda self: self._attrs)

    # --- Public interface (Dataset) ------------------------------------------

    def __init__(self, group, name, create=False,
                    data=None, dtype=None, shape=None, 
                    chunks=None, compression=None, shuffle=False, fletcher32=False):
        """ Create a new Dataset object.  There are two modes of operation:

            1.  Open an existing dataset
                If "create" is false, open an existing dataset.  An exception
                will be raised if it doesn't exist.

            2.  Create a dataset
                If "create" is True, create a new dataset.  You must supply
                *either* "data", which must be a Numpy array from which the 
                shape, dtype and initial contents will be determined, or *both* 
                "dtype" (Numpy dtype object) and "shape" (tuple of dimensions).
                Chunks/compression/shuffle/fletcher32 can also be specified.

                If a dataset of the same name already exists, creation fails.

            Creation keywords (* is default):

            chunks:        Tuple of chunk dimensions or None*
            compression:   DEFLATE (gzip) compression level, int or None*
            shuffle:       Use the shuffle filter? (requires compression) T/F*
            fletcher32:    Enable Fletcher32 error detection? T/F*
        """
        BaseNamed.__init__(self)
        if create:
            if not group._tr_active:
                self.id = h5d.py_create(group.id, name, data, shape, 
                                        chunks, compression, shuffle, fletcher32)
            else:
                action = Action("Create dataset \"%s\"" % name,
                        (h5d.py_create, (group.id, name, data, shape, chunks,
                                         compression, shuffle, fletcher32), {}),
                        (h5g.unlink, (group.id, name), {}),
                        None)
                group.manager.do(action)

        else:
            if any((data,dtype,shape,chunks,compression,shuffle,fletcher32)):
                raise ValueError('You cannot specify keywords when opening a dataset.')
            self.id = h5d.open(group.id, name)

        self._attrs = AttributeManager(self)
        self._fields = None
        self._byteorder = None
        self._string_length = None

    def __getitem__(self, *args):
        """ Read a slice from the underlying HDF5 array.  Currently only
            numerical slices are supported; for recarray-style access consider
            using the names_mask property.
        """
        if any( [isinstance(x, basestring) for x in args] ):
            raise TypeError("Slices must be numbers; recarray-style indexing is not yet supported.")

        start, count, stride = _slices_to_tuples(args)

        return h5d.py_read_slab(self.id, start, count, stride, 
                                compound_fields=self.names_mask,
                                force_native=self.force_native)

    def __setitem__(self, *args):
        """ Write to the underlying array from an existing Numpy array.  The
            shape of the Numpy array must match the shape of the selection,
            and the Numpy array's datatype must be convertible to the HDF5
            array's datatype.
        """
        val = args[-1]
        slices = args[0:len(args)-1]
        start, count, stride = _slices_to_tuples(slices)
        if not self._tr_active:
            h5d.py_write_slab(self.id, val, start, stride)
        else:
            backup = h5d.py_read_slab(self.id, start, count, stride)
            action = Action("Write slice",
                            (h5d.py_write_slab, (self.id, val, start, stride), {}),
                            (h5d.py_write_slab, (self.id, backup, start, stride), {}),
                            None )
            self.manager.do(action)

    def close(self):
        """ Force the HDF5 library to close and free this object.  You 
            shouldn't need to do this in normal operation; HDF5 objects are 
            automatically closed when their Python counterparts are deallocated.
        """
        if self._tr_active:
            raise IllegalTransactionError("close() is not a transactable operation.")
        h5d.close(self.id)

    def __del__(self):
        try:
            if self._tr_active:
                self.manager.commit()
            h5d.close(self.id)
        except:
            pass

    def __str__(self):
        return 'Dataset: '+str(self.shape)+'  '+repr(self.dtype)

    def __repr__(self):
        return self.__str__()

class Group(BaseNamed):
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

        - Setting items: See the __setitem__ docstring; the rules are:
            1. Existing Group or Dataset: create a hard link in this group
            2. Numpy array: create a new dataset here, overwriting any old one
            3. Anything else: try to create a Numpy array.  Also works with
               Python scalars which have Numpy type equivalents.

        - Deleting items: unlinks the object from this group.

        - Attribute access: through the property obj.attrs.  See the 
          AttributeManager class documentation for more information.

        - len(obj) returns the number of group members
    """

    # --- Public interface (Group) --------------------------------------------

    def __init__(self, parent_object, name, create=False):
        """ Create a new Group object, from a parent object and a name.

            If "create" is False (default), try to open the given group,
            raising an exception if it doesn't exist.  If "create" is True,
            create a new HDF5 group and link it into the parent group.
        """
        BaseNamed.__init__(self)
        self.id = 0
        if create:
            if not parent_object._tr_active:
                self.id = h5g.create(parent_object.id, name)
            else:
                action = Action('Create group "%s"' % name,
                        (h5g.create, (parent_object.id, name), {}),
                        (h5g.unlink, (parent_object.id, name), {}),
                        None)
                parent_object.manager.do(action)
        else:
            self.id = h5g.open(parent_object.id, name)
        
        #: Group attribute access (dictionary-style)
        self.attrs = AttributeManager(self)

    def __delitem__(self, name):
        """ Unlink a member from the HDF5 group.
        """
        if not self._tr_active:
            h5g.unlink(self.id, name)
        else:
            raise IllegalTransactionError("Deleting members is not (yet) a transactable operation.")

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
            objname = h5i.get_name(obj.id)
            if not self._tr_active:
                h5g.link(self.id, name, objname, link_type=h5g.LINK_HARD)
            else:
                action = Action('Create link "%s"' % name,
                            (h5g.link, (self.id, name, objname), {}),
                            (h5g.unlink, (self.id, name), {}),
                             None )
                self.manager.do(action)

        else:
            if not isinstance(obj, numpy.ndarray):
                obj = numpy.array(obj)

            if h5t.py_can_convert_dtype(obj.dtype):
                # Dataset creation is automatically recorded
                dset = Dataset(self, name, data=obj, create=True, force=True)
                dset.close()
            else:
                raise ValueError("Don't know how to store data of this type in a dataset: " + repr(obj.dtype))

    def __getitem__(self, name):
        """ Retrive the Group or Dataset object.  If the Dataset is scalar,
            returns its value instead.
        """
        retval = _open_arbitrary(self, name)
        if isinstance(retval, Dataset) and retval.shape == ():
            value = h5d.py_read_slab(retval.id, ())
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
        if not self._tr_active:
            h5g.close(self.id)
        else:
            raise IllegalTransactionError("Can't close group while transaction is active.")

    def __del__(self):
        try:
            h5g.close(self.id)
        except:
            pass

    def __str__(self):
        return 'Group (%d members): ' % self.nmembers + ', '.join(['"%s"' % name for name in self])

    def __repr__(self):
        return self.__str__()

class File(Group):

    """ Represents an HDF5 file on disk.

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
        BaseNamed.__init__(self)
        if not mode in self._modes:
            raise ValueError("Invalid mode; must be one of %s" % ', '.join(self._modes))
              
        plist = h5p.create(h5p.CLASS_FILE_ACCESS)
        try:
            h5p.set_fclose_degree(plist, h5f.CLOSE_STRONG)
            if mode == 'r':
                self.id = h5f.open(name, h5f.ACC_RDONLY, access_id=plist)
            elif 'r' in mode or 'a' in mode:
                self.id = h5f.open(name, h5f.ACC_RDWR, access_id=plist)
            elif noclobber:
                self.id = h5f.create(name, h5f.ACC_EXCL, access_id=plist)
            else:
                self.id = h5f.create(name, h5f.ACC_TRUNC, access_id=plist)
        finally:
            h5p.close(plist)

        # For __str__ and __repr__
        self.filename = name
        self.mode = mode
        self.noclobber = noclobber
        self.attrs = AttributeManager(self)

    def close(self):
        """ Close this HDF5 object.  Note that any further access to objects
            defined in this file will raise an exception.
        """
        h5f.close(self.id)

    def flush(self):
        """ Instruct the HDF5 library to flush disk buffers for this file.
        """
        h5f.flush(self.id)

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
        self.manager = None

    def __getitem__(self, name):
        obj = h5a.py_get(self.id, name)
        if len(obj.shape) == 0:
            return obj.dtype.type(obj)
        return obj

    def __setitem__(self, name, value):
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value)

        def set_attribute(objid, name, newval):
            backup = None
            if h5a.py_exists(objid, name):
                backup = h5a.py_get(objid, name)
                h5a.delete(objid, name)
            try:
                h5a.py_set(objid, name, value)
            except:
                if backup is not None:
                    h5a.py_set(objid, name, backup)
                raise

        if self.manager is not None:
            set_attribute(self.id, name, value)
        else:
            if h5a.py_exists(self.id, name):
                backup = h5a.py_get(self.id, name)
                undo = (set_attribute, (self.id, name, backup), {})
            else:
                undo = (h5a.delete, (self.id, name), {})

            action = Action("Set attribute \"%s\"" % name,
                        (set_attribute, (self.id, name, value), {}),
                        undo,
                        None)

    def __delitem__(self, name):
        if self.manager is None:
            h5a.delete(self.id, name)
        else:
            backup = h5a.py_get(self.id, name)
            action = Action("Delete \"%s\"" % name,
                        (h5a.delete, (self.id, name) ,{}),
                        (h5a.py_set, (self.id, name, backup), {}),
                        None)

    def __len__(self):
        return h5a.get_num_attrs(self.id)

    def __iter__(self):
        return h5a.py_listattrs(self.id)

    def iteritems(self):
        for name in self:
            yield (name, self[name])

    def __str__(self):
        return "Attributes: "+', '.join(['"%s"' % x for x in self])

class NamedType(BaseNamed):

    """ Represents a named datatype, stored in a file.  

        HDF5 datatypes are typically represented by their Numpy dtype
        equivalents; this class exists mainly to provide access to attributes
        stored on HDF5 named types.  Properties:

        dtype:   Equivalent Numpy dtype for this HDF5 type
        attrs:   AttributeManager instance for attribute access

        Mutating the returned dtype object has no effect on the underlying
        HDF5 datatype.
    """ 
        
    def _get_dtype(self):
        if self._dtype is None:
            self._dtype = h5t.py_h5t_to_dtype(self.id)
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
            tid = h5t.py_dtype_to_h5t(dtype)
            try:
                h5t.commit(group.id, name, tid)
            finally:
                h5t.close(tid)

        self.id = h5t.open(group.id, name)
        self.attrs = AttributeManager(self)

    def close(self):
        """ Force the library to close this object.  Not ordinarily required.
        """
        if self.id is not None:
            h5t.close(self.id)

    def __del__(self):
        if self.id is not None:
            try:
                h5t.close(self.id)
            except H5Error:
                pass
    


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
            if type_code == h5g.OBJ_GROUP:
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
                    and h5g.get_objinfo(self.group.id,x).type == h5g.OBJ_GROUP]

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

    if info.type == h5g.OBJ_GROUP:      # group
        return Group(group_obj, name)

    elif info.type == h5g.OBJ_DATASET:  # dataset
        return Dataset(group_obj, name)

    elif info.type == h5g.OBJ_DATATYPE: # named type
        return NamedDatatype(group_obj, name)

    raise NotImplementedError('Object type "%s" unsupported by the high-level interface.' % h5g.OBJ_MAPPER[info.type])

def _slices_to_tuples(args):
    """ Turns a series of slice objects into the start, count, stride tuples
        expected by py_read/py_write
    """

    startlist = []
    countlist = []
    stridelist = []
    
    if len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]

    for arg in args:

        if isinstance(arg, slice):

            if arg.start is None:
                start=0
            else:
                if arg.start < 0:
                    raise ValueError("Negative dimensions are not allowed")
                start=arg.start

            if arg.step is None:
                step = 1
            else:
                if arg.step < 0:
                    raise ValueError("Negative step sizes are not allowed")
                step = arg.step

            startlist.append(start)
            stridelist.append(step)

            if arg.stop is None:
                countlist.append(None)
            else:
                if arg.stop < 0:
                    raise ValueError("Negative dimensions are not allowed")
                count = (arg.stop-start)/step
                if count == 0:
                    raise ValueError("Zero-length selections are not allowed")
                countlist.append(count)

        else:
            startlist.append(arg)
            countlist.append(1)
            stridelist.append(1)

    return (tuple(startlist), tuple(countlist), tuple(stridelist))

#: Command-line HDF5 file "shell": browse(name) (or browse() for last file).
browse = _H5Browse()













