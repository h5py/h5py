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

    Objects in this module are designed to provide a friendly, Python-style
    interface to native HDF5 concepts like files, datasets, groups and
    attributes.  The module is written in pure Python and uses the standard
    h5py low-level interface exclusively.

    Most components defined here are re-exported into the root h5py package
    namespace, because they are the most straightforward and intuitive
    way to interact with HDF5.
"""

from __future__ import with_statement

import os
import numpy
import warnings
import sys
import math

import os.path as op
import posixpath as pp

from h5py import h5, h5f, h5g, h5s, h5t, h5d, h5a, \
                 h5p, h5r, h5z, h5i, h5fd, h5o, h5l, \
                 version, filters
import h5py.selections as sel

config = h5.get_config()

def _memo_property(meth):
    """ Convenience decorator for memoized properties.

    Intended for read-only, unchanging properties.  Instead of caching values
    on the instance directly (i.e. self._value = value), stores in a weak-key
    dictionary as dct[self] = value.

    In addition to not polluting the instance dict, it provides a way to cache
    values across instances; any two instances which hash to the same value
    and compare equal will return the same value when the property is read.
    This allows the sharing of things like file modes and per-file locks,
    which are tied to the underlying file and not any particular instance.

    Caveats:
    1. A strong reference is held to the value, so returning self is a bad idea
    2. Can't initialize the value in a constructor, unlike self._value caching
    """
    import functools
    import weakref
    dct = weakref.WeakKeyDictionary()
    def wrap(self):
        if self not in dct:
            return dct.setdefault(self, meth(self))
        return dct[self]
    functools.update_wrapper(wrap, meth)
    return property(wrap)

def _hbasename(name):
    """ Basename function with more readable handling of trailing slashes"""
    name = pp.basename(pp.normpath(name))
    return name if name != '' else '/'

def _hsizestring(size):
    """ Friendly representation of byte sizes """
    d = int(math.log(size, 1024) // 1) if size else 0
    suffix = {1: 'k', 2: 'M', 3: 'G', 4: 'T'}.get(d)
    if suffix is None:
        return "%d bytes" % size
    return "%.1f%s" % (size / (1024.**d), suffix)
    
def is_hdf5(fname):
    """ Determine if a file is valid HDF5 (False if it doesn't exist). """
    fname = os.path.abspath(fname)

    if os.path.isfile(fname):
        try:
            fname = fname.encode(sys.getfilesystemencoding())
        except (UnicodeError, LookupError):
            pass
        return h5f.is_hdf5(fname)
    return False

# === Base classes ============================================================

class HLObject(object):

    """
        Base class for high-level interface objects.

        All objects of this class support the following properties:

        id:     Low-level identifer, compatible with the h5py.h5* modules.
        name:   Name of this object in the HDF5 file.  May not be unique.
        attrs:  HDF5 attributes of this object.  See AttributeManager class.
        file:   The File instance associated with this object
        parent: (A) parent of this object, according to dirname(obj.name)
        ref:    An HDF5 reference to this object.

        Equality comparison and hashing are based on native HDF5 object
        identity.
    """

    @property
    def name(self):
        """Name of this object in the HDF5 file.  Not necessarily unique."""
        name = h5i.get_name(self.id)
        if name is None and config.API_18:
            name = h5r.get_name(self.ref)
        return name

    @_memo_property
    def attrs(self):
        """Provides access to HDF5 attributes. See AttributeManager."""
        return AttributeManager(self)

    @_memo_property
    def _file(self):
        fid = h5i.get_file_id(self.id)
        return File(None, bind=fid)
 
    @property
    def file(self):
        """Return a File instance associated with this object"""
        if isinstance(self, File):
            return self
        return self._file

    @property
    def parent(self):
        """Return the parent group of this object.

        This is always equivalent to file[posixpath.basename(obj.name)].
        """
        if self.name is None:
            raise ValueError("Parent of an anonymous object is undefined")
        return self.file[pp.dirname(self.name)]

    @property
    def ref(self):
        """ An (opaque) HDF5 reference to this object """
        return h5r.create(self.id, '.', h5r.OBJECT)

    @property
    def _lock(self):
        return self.file._lock

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
    
    def keys(self):
        """ Get a list containing member names """
        with self._lock:
            return list(self)

    def iterkeys(self):
        """ Get an iterator over member names """
        with self._lock:
            return iter(self)

    def values(self):
        """ Get a list containing member objects """
        with self._lock:
            return [self[x] for x in self]

    def itervalues(self):
        """ Get an iterator over member objects """
        with self._lock:
            for x in self:
                yield self[x]

    def items(self):
        """ Get a list of tuples containing (name, object) pairs """
        with self._lock:
            return [(x, self[x]) for x in self]

    def iteritems(self):
        """ Get an iterator over (name, object) pairs """
        with self._lock:
            for x in self:
                yield (x, self[x])

    def get(self, name, default=None):
        """ Retrieve the member, or return default if it doesn't exist """
        with self._lock:
            if name in self:
                return self[name]
            return default

    # Compatibility methods
    def listnames(self):
        """ Deprecated alias for keys() """
        warnings.warn("listnames() is deprecated; use keys() instead", DeprecationWarning)
        return self.keys()
    def iternames(self):
        """ Deprecated alias for iterkeys() """
        warnings.warn("iternames() is deprecated; use iterkeys() instead", DeprecationWarning)
        return self.iterkeys()
    def listobjects(self):
        """ Deprecated alias for values() """
        warnings.warn("listobjects() is deprecated; use values() instead", DeprecationWarning)
        return self.values()
    def iterobjects(self):
        """ Deprecated alias for itervalues() """
        warnings.warn("iterobjects() is deprecated; use itervalues() instead", DeprecationWarning)
        return self.itervalues()
    def listitems(self):
        """ Deprecated alias for items() """
        warnings.warn("listitems() is deprecated; use items() instead", DeprecationWarning)
        return self.items()

class Group(HLObject, _DictCompat):

    """ Represents an HDF5 group.

        It's recommended to use the Group/File method create_group to create
        these objects, rather than trying to create them yourself.

        Groups implement a basic dictionary-style interface, supporting
        __getitem__, __setitem__, __len__, __contains__, keys(), values()
        and others.

        They also contain the necessary methods for creating new groups and
        datasets.  Group attributes can be accessed via <group>.attrs.
    """

    def __init__(self, parent_object, name, create=False, _rawid=None):
        """ Create a new Group object, from a parent object and a name.

        If "create" is False (default), try to open the given group,
        raising an exception if it doesn't exist.  If "create" is True,
        create a new HDF5 group and link it into the parent group.

        It's recommended to use __getitem__ or create_group() rather than
        calling the constructor directly.
        """
        with parent_object._lock:
            if _rawid is not None:
                self.id = _rawid
            elif create:
                self.id = h5g.create(parent_object.id, name)
            else:
                self.id = h5g.open(parent_object.id, name)
    
    def __setitem__(self, name, obj):
        """ Add an object to the group.  The name must not already be in use.

        The action taken depends on the type of object assigned:

        Named HDF5 object (Dataset, Group, Datatype)
            A hard link is created at "name" which points to the
            given object.

        SoftLink or ExternalLink
            Create the corresponding link.

        Numpy ndarray
            The array is converted to a dataset object, with default
            settings (contiguous storage, etc.).

        Numpy dtype
            Commit a copy of the datatype as a named datatype in the file.

        Anything else
            Attempt to convert it to an ndarray and store it.  Scalar
            values are stored as scalar datasets. Raise ValueError if we
            can't understand the resulting array dtype.
        """
        with self._lock:
            if isinstance(obj, Group) or isinstance(obj, Dataset) or isinstance(obj, Datatype):
                self.id.link(h5i.get_name(obj.id), name, link_type=h5g.LINK_HARD)

            elif isinstance(obj, SoftLink):
                self.id.link(obj.path, name, link_type=h5g.LINK_SOFT)
    
            elif isinstance(obj, ExternalLink):
                self.id.links.create_external(name, obj.filename, obj.path)

            elif isinstance(obj, numpy.dtype):
                htype = h5t.py_create(obj)
                htype.commit(self.id, name)

            else:
                self.create_dataset(name, data=obj)

    def __getitem__(self, name):
        """ Open an object attached to this group. 
        """
        with self._lock:

            if isinstance(name, h5r.Reference):

                if not name:
                    raise ValueError("Empty reference")
                kind = h5r.get_obj_type(name, self.id)
                if kind == h5g.GROUP:
                    return Group(self, None, _rawid=h5r.dereference(name, self.id))
                elif kind == h5g.DATASET:
                    return Dataset(self, None, _rawid=h5r.dereference(name, self.id))

                raise ValueError("Unrecognized reference object type")

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
        """ Create and return a subgroup. Fails if the group already exists.
        """
        return Group(self, name, create=True)

    def require_group(self, name):
        """ Check if a group exists, and create it if not.  TypeError if an
        incompatible object exists.
        """
        with self._lock:
            if not name in self:
                return self.create_group(name)
            grp = self[name]
            if not isinstance(grp, Group):
                raise TypeError("Incompatible object (%s) already exists" % grp.__class__.__name__)
            return grp

    def create_dataset(self, name, *args, **kwds):
        """ Create and return a new dataset.  Fails if "name" already exists.

        create_dataset(name, shape, [dtype=<Numpy dtype>], **kwds)
        create_dataset(name, data=<Numpy array>, **kwds)

        The default dtype is '=f4' (single-precision float).

        Additional keywords ("*" is default):

        chunks
            Tuple of chunk dimensions or None*

        maxshape
            None* or a tuple giving maximum dataset size.  An element of None
            indicates an unlimited dimension.  Dataset can be expanded by
            calling resize()

        compression
            Compression strategy; None*, 'gzip', 'szip' or 'lzf'.  An integer
            is interpreted as a gzip level.

        compression_opts
            Optional compression settings; for gzip, this may be an int.  For
            szip, it should be a 2-tuple ('ec'|'nn', int(0-32)).   

        shuffle
            Use the shuffle filter (increases compression performance for
            gzip and LZF).  True/False*.

        fletcher32
            Enable error-detection.  True/False*.
        """
        return Dataset(self, name, *args, **kwds)

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        """Open a dataset, or create it if it doesn't exist.

        Checks if a dataset with compatible shape and dtype exists, and
        creates one if it doesn't.  Raises TypeError if an incompatible
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
                raise TypeError("Incompatible object (%s) already exists" % dset.__class__.__name__)

            if not shape == dset.shape:
                raise TypeError("Shapes do not match (existing %s vs new %s)" % (dset.shape, shape))

            if exact:
                if not dtype == dset.dtype:
                    raise TypeError("Datatypes do not exactly match (existing %s vs new %s)" % (dset.dtype, dtype))
            elif not numpy.can_cast(dtype, dset.dtype):
                raise TypeError("Datatypes cannot be safely cast (existing %s vs new %s)" % (dset.dtype, dtype))
            
            return dset

    def get(self, name, default=None, getclass=False, getlink=False):
        """ Retrieve item "name", or "default" if it's not in this group.

        getclass
            If True, returns the class of object (Group, Dataset, etc.)
            instead of the object itself.

        getlink
            If True, return SoftLink and ExternalLink instances instead
            of the objects they point to.
        """
        with self._lock:

            if not name in self:
                return default

            if config.API_18:

                linkinfo = self.id.links.get_info(name)

                if linkinfo.type == h5l.TYPE_HARD or not getlink:

                    objinfo = h5o.get_info(self.id, name)
                    cls = {h5o.TYPE_GROUP: Group, h5o.TYPE_DATASET: Dataset,
                           h5o.TYPE_NAMED_DATATYPE: Datatype}.get(objinfo.type)
                    if cls is None:
                        raise TypeError("Unknown object type")

                    return cls if getclass else cls(self, name)

                else:
                    if linkinfo.type == h5l.TYPE_SOFT:
                        return SoftLink if getclass else SoftLink(self.id.links.get_val(name))
                    elif linkinfo.type == h5l.TYPE_EXTERNAL:
                        return ExternalLink if getclass else ExternalLink(*self.id.links.get_val(name))

                    raise TypeError("Unknown link class")

            # API 1.6
            info = h5g.get_objinfo(self.id, name, follow_link=(not getlink))

            cls = {h5g.DATASET: Dataset, h5g.GROUP: Group,
                   h5g.TYPE: Datatype}.get(info.type)

            if cls is not None:
                return cls if getclass else cls(self, name)

            if getlink and info.type == h5g.LINK:
                return SoftLink if getclass else SoftLink(self.id.get_linkval(name))
                    
            raise TypeError("Unknown object type")

    # New 1.8.X methods

    def copy(self, source, dest, name=None):
        """ Copy an object or group (Requires HDF5 1.8).

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
        """ Recursively visit all names in this group and subgroups (HDF5 1.8).

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
        """
        if not config.API_18:
            raise NotImplementedError("This feature is only available with HDF5 1.8.0 and later")
    
        with self._lock:
            return h5o.visit(self.id, func)

    def visititems(self, func):
        """ Recursively visit names and objects in this group (HDF5 1.8).

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
                namestr = '"%s"' % self.name if self.name is not None else "(anonymous)"
                return '<HDF5 group %s (%d members)>' % \
                    (namestr, len(self))
            except Exception:
                return "<Closed HDF5 group>"

class File(Group):

    """ Represents an HDF5 file on disk.

        File(name, mode=None, driver=None, **driver_kwds)

        Legal modes: r, r+, w, w-, a (default)

        File objects inherit from Group objects; Group-like methods all
        operate on the HDF5 root group ('/').  Like Python file objects, you
        must close the file ("obj.close()") when you're done with it. File
        objects may also be used as context managers in Python "with" blocks.

        The HDF5 file driver may also be specified:

        None
            Use the standard HDF5 driver appropriate for the current platform.
            On UNIX, this is the H5FD_SEC2 driver; on Windows, it is
            H5FD_WINDOWS.

        'sec2'
            Unbuffered, optimized I/O using standard POSIX functions.

        'stdio' 
            Buffered I/O using functions from stdio.h.

        'core'
            Memory-map the entire file; all operations are performed in
            memory and written back out when the file is closed.  Keywords:

            backing_store:  If True (default), save changes to a real file
                            when closing.  If False, the file exists purely
                            in memory and is discarded when closed.

            block_size:     Increment (in bytes) by which memory is extended.
                            Default is 1 megabyte (1024**2).

        'family'
            Store the file on disk as a series of fixed-length chunks.  Useful
            if the file system doesn't allow large files.  Note: the filename
            you provide *must* contain a printf-style integer format code
            (e.g. %d"), which will be replaced by the file sequence number.
            Keywords:

            memb_size:  Maximum file size (default is 2**31-1).
    """

    @_memo_property
    def _lock(self):
        """ Get an RLock for this file, creating it if necessary.  Locks are
        linked to the "real" underlying HDF5 file, regardless of the number
        of File instances.
        """
        import threading
        return threading.RLock()

    @property
    def filename(self):
        """File name on disk"""
        name = h5f.get_name(self.fid)
        # Note the exception can happen in one of two ways:
        # 1. The name doesn't comply with the file system encoding;
        #    return the raw byte string
        # 2. The name can't be encoded down to ASCII; return it as
        #    a Unicode string object
        try:
            name = name.decode(sys.getfilesystemencoding())
            return name.encode('ascii')
        except (UnicodeError, LookupError):
            return name

    @_memo_property
    def mode(self):
        """Python mode used to open file"""
        if hasattr(self, '_mode'):
            return self._mode
        if not config.API_18:
            return None
        intent = self.fid.get_intent()
        return {h5f.ACC_RDONLY: 'r', h5f.ACC_RDWR: 'r+'}.get(intent)

    @property
    def driver(self):
        """Low-level HDF5 file driver used to open file"""
        drivers = {h5fd.SEC2: 'sec2', h5fd.STDIO: 'stdio',
                   h5fd.CORE: 'core', h5fd.FAMILY: 'family',
                   h5fd.WINDOWS: 'windows'}
        return drivers.get(self.fid.get_access_plist().get_driver(), 'unknown')

    # --- Public interface (File) ---------------------------------------------

    def __init__(self, name, mode=None, driver=None, **kwds):
        """ Create a new file object.  

        Valid modes (like Python's file() modes) are: 
        - r   Readonly, file must exist
        - r+  Read/write, file must exist
        - w   Create file, truncate if exists
        - w-  Create file, fail if exists
        - a   Read/write if exists, create otherwise (default)

        Valid drivers are:
        - None      Use default driver ('sec2' on UNIX, 'windows' on Win32) 
        - 'sec2'    Standard UNIX driver
        - 'stdio'   Stdio (buffered) driver
        - 'core'    mmap driver
        - 'family'  Multi-part file driver
        """
        if "bind" in kwds:
            self.fid = kwds["bind"]
        else:
            if driver == 'core' and mode=='w-' and version.hdf5_version_tuple[0:2] == (1,6):
                raise NotImplementedError("w- flag does not work on 1.6 for CORE driver")
            try:
                # If the byte string doesn't match the default encoding, just
                # pass it on as-is.  Note Unicode objects can always be encoded.
                name = name.encode(sys.getfilesystemencoding())
            except (UnicodeError, LookupError):
                pass

            plist = self._get_access_plist(driver, **kwds)
            self.fid = self._get_fid(name, mode, plist)
            self._mode = mode

        self.id = self.fid  # So the Group constructor can find it.
        Group.__init__(self, self, '/')

    def _get_access_plist(self, driver, **kwds):
        """ Set up file access property list """
        plist = h5p.create(h5p.FILE_ACCESS)
        plist.set_fclose_degree(h5f.CLOSE_STRONG)

        if driver is None or (driver=='windows' and sys.platform=='win32'):
            return plist

        if(driver=='sec2'):
            plist.set_fapl_sec2(**kwds)
        elif(driver=='stdio'):
            plist.set_fapl_stdio(**kwds)
        elif(driver=='core'):
            plist.set_fapl_core(**kwds)
        elif(driver=='family'):
            plist.set_fapl_family(memb_fapl=plist.copy(), **kwds)
        else:
            raise ValueError('Unknown driver type "%s"' % driver)

        return plist

    def _get_fid(self, name, mode, plist):
        """ Get a new FileID by opening or creating a file.
        Also validates mode argument."""
        if mode == 'r':
            fid = h5f.open(name, h5f.ACC_RDONLY, fapl=plist)
        elif mode == 'r+':
            fid = h5f.open(name, h5f.ACC_RDWR, fapl=plist)
        elif mode == 'w-':
            fid = h5f.create(name, h5f.ACC_EXCL, fapl=plist)
        elif mode == 'w':
            fid = h5f.create(name, h5f.ACC_TRUNC, fapl=plist)
        elif mode == 'a' or mode is None:
            try:
                fid = h5f.open(name, h5f.ACC_RDWR, fapl=plist)
            except IOError:
                fid = h5f.create(name, h5f.ACC_EXCL, fapl=plist)
        else:
            raise ValueError("Invalid mode; must be one of r, r+, w, w-, a")
        return fid

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
                return '<HDF5 file "%s" (mode %s, %d bytes)>' % \
                    (os.path.basename(self.filename), self.mode,
                     _hsizestring(self.id.get_filesize()))
            except Exception:
                return "<Closed HDF5 file>"

    # Fix up identity to use the file identifier, not the root group.
    def __hash__(self):
        return hash(self.fid)
    def __eq__(self, other):
        if hasattr(other, 'fid'):
            return self.fid == other.fid
        return False

class _RegionProxy(object):

    def __init__(self, dset):
        self.id = dset.id

    def __getitem__(self, args):
        
        selection = sel.select(self.id.shape, args, dsid=self.id)
        return h5r.create(self.id, '.', h5r.DATASET_REGION, selection._id)

class Dataset(HLObject):

    """ High-level interface to an HDF5 dataset.

        Datasets can be opened via the syntax Group[<dataset name>], and
        created with the method Group.create_dataset().

        Datasets behave superficially like Numpy arrays.  NumPy "simple"
        slicing is fully supported, along with a subset of fancy indexing
        and indexing by field names (dataset[0:10, "fieldname"]).

        The standard NumPy properties "shape" and "dtype" are also available.
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
        """  Deprecated alias for dataset[...] and dataset[()] """
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
        with self._lock:
            space = self.id.get_space()
            dims = space.get_simple_extent_dims(True)
            return tuple(x if x != h5s.UNLIMITED else None for x in dims)

    @_memo_property
    def regionref(self):
        return _RegionProxy(self)

    def __init__(self, group, name,
                    shape=None, dtype=None, data=None,
                    chunks=None, compression=None, shuffle=None,
                    fletcher32=None, maxshape=None, compression_opts=None,
                    _rawid = None):
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
            if _rawid is not None:
                self.id = _rawid
            elif data is None and shape is None:
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
                type_id = h5t.py_create(dtype, logical=True)

                self.id = h5d.create(group.id, name, type_id, space_id, plist)
                if data is not None:
                    self.id.write(h5s.ALL, h5s.ALL, data)

            plist = self.id.get_create_plist()
            self._filters = filters.get_filters(plist)
            if plist.get_layout() == h5d.CHUNKED:
                self._chunks = plist.get_chunk()
            else:
                self._chunks = None

    def resize(self, size, axis=None):
        """ Resize the dataset, or the specified axis (HDF5 1.8 only).

        The dataset must be stored in chunked format; it can be resized up to
        the "maximum shape" (keyword maxshape) specified at creation time.
        The rank of the dataset cannot be changed.

        "Size" should be a shape tuple, or if an axis is specified, an integer.

        BEWARE: This functions differently than the NumPy resize() method!
        The data is not "reshuffled" to fit in the new shape; each axis is
        grown or shrunk independently.  The coordinates of existing data is
        fixed.
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

        Limited to 2**32 on 32-bit systems; Dataset.len() is preferred.
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
        """ Iterate over the first axis.  TypeError if scalar.

        BEWARE: Modifications to the yielded data are *NOT* written to file.
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
            selection = sel.select(self.shape, args, dsid=self.id)

            if selection.nselect == 0:
                return numpy.ndarray((0,), dtype=new_dtype)

            # Create the output array using information from the selection.
            arr = numpy.ndarray(selection.mshape, new_dtype, order='C')

            # This is necessary because in the case of array types, NumPy
            # discards the array information at the top level.
            mtype = h5t.py_create(new_dtype)

            # Perfom the actual read
            mspace = h5s.create_simple(selection.mshape)
            fspace = selection._id
            self.id.read(mspace, fspace, arr, mtype)

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

        Classes from the "selections" module may also be used to index.
        """
        with self._lock:

            args = args if isinstance(args, tuple) else (args,)

            # Sort field indices from the slicing
            names = tuple(x for x in args if isinstance(x, str))
            args = tuple(x for x in args if not isinstance(x, str))

            if len(names) != 0:
                raise TypeError("Field name selections are not allowed for write.")

            # Generally we try to avoid converting the arrays on the Python
            # side.  However, for compound literals this is unavoidable.
            if self.dtype.kind == 'V' and \
            (not isinstance(val, numpy.ndarray) or val.dtype.kind != 'V'):
                val = numpy.asarray(val, dtype=self.dtype, order='C')
            else:
                val = numpy.asarray(val, order='C')

            # Check for array dtype compatibility and convert
            if self.dtype.subdtype is not None:
                shp = self.dtype.subdtype[1]
                if val.shape[-len(shp):] != shp:
                    raise TypeError("Can't broadcast to array dimension %s" % (shp,))
                mtype = h5t.py_create(numpy.dtype((val.dtype, shp)))
                mshape = val.shape[0:len(val.shape)-len(shp)]
            else:
                mshape = val.shape
                mtype = None

            # Perform the dataspace selection
            selection = sel.select(self.shape, args, dsid=self.id)

            if selection.nselect == 0:
                return

            # Broadcast scalars if necessary.
            if (mshape == () and selection.mshape != ()):
                if self.dtype.subdtype is not None:
                    raise NotImplementedError("Scalar broadcasting is not supported for array dtypes")
                val2 = numpy.empty(selection.mshape[-1], dtype=val.dtype)
                val2[...] = val
                val = val2
                mshape = val.shape

            # Perform the write, with broadcasting
            # Be careful to pad memory shape with ones to avoid HDF5 chunking
            # glitch, which kicks in for mismatched memory/file selections
            if(len(mshape) < len(self.shape)):
                mshape_pad = (1,)*(len(self.shape)-len(mshape)) + mshape
            else:
                mshape_pad = mshape
            mspace = h5s.create_simple(mshape_pad, (h5s.UNLIMITED,)*len(mshape_pad))
            for fspace in selection.broadcast(mshape):
                self.id.write(mspace, fspace, val, mtype)

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
            source_sel = sel.select(self.shape, source_sel, self.id)  # for numpy.s_
        fspace = source_sel._id

        if dest_sel is None:
            dest_sel = sel.SimpleSelection(dest.shape)
        else:
            dest_sel = sel.select(dest.shape, dest_sel, self.id)

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
            source_sel = sel.select(source.shape, source_sel, self.id)  # for numpy.s_
        mspace = source_sel._id

        if dest_sel is None:
            dest_sel = sel.SimpleSelection(self.shape)
        else:
            dest_sel = sel.select(self.shape, dest_sel, self.id)

        for fspace in dest_sel.broadcast(source_sel.mshape):
            self.id.write(mspace, fspace, source)

    def __repr__(self):
        with self._lock:
            try:
                namestr = '"%s"' % _hbasename(self.name) if self.name is not None else "(anonymous)"
                return '<HDF5 dataset %s: shape %s, type "%s">' % \
                    (namestr, self.shape, self.dtype.str)
            except Exception:
                return "<Closed HDF5 dataset>"

class AttributeManager(_DictCompat):

    """ Allows dictionary-style access to an HDF5 object's attributes.

        These are created exclusively by the library and are available as
        a Python attribute at <object>.attrs

        Like the members of groups, attributes provide a minimal dictionary-
        style interface.  Anything which can be reasonably converted to a
        Numpy array or Numpy scalar can be stored.

        Attributes are automatically created on assignment with the
        syntax <obj>.attrs[name] = value, with the HDF5 type automatically
        deduced from the value.  Existing attributes are overwritten.

        To modify an existing attribute while preserving its type, use the
        method modify().  To specify an attribute of a particular type and
        shape (or to create an empty attribute), use create().
    """

    def __init__(self, parent):
        """ Private constructor.
        """
        self.id = parent.id
        self._file = parent.file

    @property
    def _lock(self):
        return self._file._lock

    def __getitem__(self, name):
        """ Read the value of an attribute.
        """
        with self._lock:
            attr = h5a.open(self.id, name)

            arr = numpy.ndarray(attr.shape, dtype=attr.dtype, order='C')
            attr.read(arr)

            if len(arr.shape) == 0:
                return numpy.asscalar(arr)
            return arr

    def __setitem__(self, name, value):
        """ Set a new attribute, overwriting any existing attribute.

        The type and shape of the attribute are determined from the data.  To
        use a specific type or shape, or to preserve the type of an attribute,
        use the methods create() and modify().

        Broadcasting isn't supported for attributes.
        """
        with self._lock:
            self.create(name, data=value)

    def __delitem__(self, name):
        """ Delete an attribute (which must already exist). """
        h5a.delete(self.id, name)

    def create(self, name, data, shape=None, dtype=None):
        """ Create a new attribute, overwriting any existing attribute.

        name:   Name of the new attribute (required)
        data:   An array to initialize the attribute (required)
        shape:  Shape of the attribute.  Overrides data.shape if both are
                given.  The total number of points must be unchanged.
        dtype:  Data type of the attribute.  Overrides data.dtype if both
                are given.  Must be conversion-compatible with data.dtype.
        """
        with self._lock:
            if data is not None:
                data = numpy.asarray(data, order='C', dtype=dtype)
                if shape is None:
                    shape = data.shape
                elif numpy.product(shape) != numpy.product(data.shape):
                    raise ValueError("Shape of new attribute conflicts with shape of data")
                    
                if dtype is None:
                    dtype = data.dtype

            if dtype is None:
                dtype = numpy.dtype('f')
            if shape is None:
                raise ValueError('At least one of "shape" or "data" must be given')

            data = data.reshape(shape)

            space = h5s.create_simple(shape)
            htype = h5t.py_create(dtype, logical=True)

            if name in self:
                h5a.delete(self.id, name)

            attr = h5a.create(self.id, name, htype, space)
            if data is not None:
                attr.write(data)

    def modify(self, name, value):
        """ Change the value of an attribute while preserving its type.

        Differs from __setitem__ in that the type of an existing attribute
        is preserved.  Useful for interacting with externally generated files.

        If the attribute doesn't exist, it will be automatically created.
        """
        with self._lock:
            if not name in self:
                self[name] = value
            else:
                value = numpy.asarray(value, order='C')

                attr = h5a.open(self.id, name)

                # Allow the case of () <-> (1,)
                if (value.shape != attr.shape) and not \
                   (numpy.product(value.shape)==1 and numpy.product(attr.shape)==1):
                    raise TypeError("Shape of data is incompatible with existing attribute")
                attr.write(value)

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
                namestr = '"%s"' % _hbasename(self.id.name) if self.id.name is not None else "(anonymous)"
                return '<Attributes of HDF5 object %s (%d)>' % \
                    (namestr, len(self))
            except Exception:
                return "<Attributes of closed HDF5 object>"


class Datatype(HLObject):

    """
        Represents an HDF5 named datatype stored in a file.

        To store a datatype, simply assign it to a name in a group:

        >>> MyGroup["name"] = numpy.dtype("f")
        >>> named_type = MyGroup["name"]
        >>> assert named_type.dtype == numpy.dtype("f")
    """

    @property
    def dtype(self):
        """Numpy dtype equivalent for this datatype"""
        return self.id.dtype

    def __init__(self, grp, name):
        """ Private constructor.
        """
        with grp._lock:
            self.id = h5t.open(grp.id, name)

    def __repr__(self):
        with self._lock:
            try:
                namestr = '"%s"' % _hbasename(self.name) if self.name is not None else "(anonymous)"
                return '<HDF5 named type "%s" (dtype %s)>' % \
                    (namestr, self.dtype.str)
            except Exception:
                return "<Closed HDF5 named type>"

class SoftLink(object):

    """
        Represents a symbolic ("soft") link in an HDF5 file.  The path
        may be absolute or relative.  No checking is performed to ensure
        that the target actually exists.
    """

    @property
    def path(self):
        return self._path

    def __init__(self, path):
        self._path = str(path)

    def __repr__(self):
        return '<SoftLink to "%s">' % self.path

class ExternalLink(object):

    """
        Represents an HDF5 external link.  Paths may be absolute or relative.
        No checking is performed to ensure either the target or file exists.
    """

    @property
    def path(self):
        return self._path

    @property
    def filename(self):
        return self._filename

    def __init__(self, filename, path):
        if not config.API_18:
            raise NotImplementedError("External links are only available as of HDF5 1.8")
        self._filename = str(filename)
        self._path = str(path)

    def __repr__(self):
        return '<ExternalLink to "%s" in file "%s"' % (self.path, self.filename)

