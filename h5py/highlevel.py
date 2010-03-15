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
import sys
import weakref
import threading
import warnings
import os.path as op
import posixpath as pp
import numpy

from h5py import h5, h5f, h5g, h5s, h5t, h5d, h5a, \
                 h5p, h5r, h5z, h5i, h5fd, h5o, h5l, \
                 version, filters, _extras
import h5py.selections as sel

from h5py.h5e import register_thread

config = h5.get_config()
phil = threading.RLock()

def is_hdf5(fname):
    """ Determine if a file is valid HDF5 (False if it doesn't exist). """
    register_thread()
    fname = os.path.abspath(fname)

    if os.path.isfile(fname):
        try:
            fname = fname.encode(sys.getfilesystemencoding())
        except (UnicodeError, LookupError):
            pass
        return h5f.is_hdf5(fname)
    return False

def _guess_dtype(data):
    """ Attempt to guess an appropriate dtype for the object, returning None
    if nothing is appropriate (or if it should be left up the the array
    constructor to figure out)
    """
    if isinstance(data, h5r.RegionReference):
        return h5t.special_dtype(ref=h5r.RegionReference)
    if isinstance(data, h5r.Reference):
        return h5t.special_dtype(ref=h5r.Reference)
    return None

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
    def file(self):
        """Return a File instance associated with this object"""
        register_thread()
        fid = h5i.get_file_id(self.id)
        return File(None, bind=fid)

    @property
    def _lapl(self):
        """Default link access property list (1.8)"""

        lapl = h5p.create(h5p.LINK_ACCESS)
        fapl = h5p.create(h5p.FILE_ACCESS)
        fapl.set_fclose_degree(h5f.CLOSE_STRONG)
        lapl.set_elink_fapl(fapl)
        return lapl

    @property
    def _lcpl(self):
        """Default link creation property list (1.8)"""
        lcpl = h5p.create(h5p.LINK_CREATE)
        lcpl.set_create_intermediate_group(True)
        return lcpl

    @property
    def id(self):
        """ Low-level identifier appropriate for this object """
        return self._id

    @property
    def name(self):
        """Name of this object in the HDF5 file.  Not necessarily unique."""
        register_thread()
        return h5i.get_name(self.id)

    @_extras.cproperty('_attrs')
    def attrs(self):
        """Provides access to HDF5 attributes. See AttributeManager."""
        return AttributeManager(self)

    @property
    def parent(self):
        """Return the parent group of this object.

        This is always equivalent to file[posixpath.basename(obj.name)].
        """
        if self.name is None:
            raise ValueError("Parent of an anonymous object is undefined")
        return self.file[pp.dirname(self.name)]

    @_extras.cproperty('_ref')
    def ref(self):
        """ An (opaque) HDF5 reference to this object """
        register_thread()
        return h5r.create(self.id, '.', h5r.OBJECT)

    def __init__(self, oid):
        """ Setup this object, given its low-level identifier """
        #self._file = self._get_file(oid)
        self._id = oid

    def __nonzero__(self):
        register_thread()
        return self.id.__nonzero__()

    def __hash__(self):
        register_thread()
        return hash(self.id)
    def __eq__(self, other):
        register_thread()
        if hasattr(other, 'id'):
            return self.id == other.id
        return False
    def __ne__(self, other):
        return not self.__eq__(other)

class _DictCompat(object):

    """
        Contains dictionary-style compatibility methods for groups and
        attributes.
    """
    
    def keys(self):
        """ Get a list containing member names """
        with phil:
            return list(self)

    def iterkeys(self):
        """ Get an iterator over member names """
        with phil:
            return iter(self)

    def values(self):
        """ Get a list containing member objects """
        with phil:
            return [self[x] for x in self]

    def itervalues(self):
        """ Get an iterator over member objects """
        with phil:
            for x in self:
                yield self[x]

    def items(self):
        """ Get a list of tuples containing (name, object) pairs """
        with phil:
            return [(x, self[x]) for x in self]

    def iteritems(self):
        """ Get an iterator over (name, object) pairs """
        with phil:
            for x in self:
                yield (x, self[x])

    def get(self, name, default=None):
        """ Retrieve the member, or return default if it doesn't exist """
        with phil:
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
        register_thread()
        with phil:
            if _rawid is not None:
                id = _rawid
            elif create:
                id = h5g.create(parent_object.id, name)
            else:
                id = h5g.open(parent_object.id, name)
            HLObject.__init__(self, id)

    def _set18(self, name, obj):
        """ HDF5 1.8 __setitem__.  PHIL should already be held. 

        Distinct from 1.6 version in that it uses the proper link creation
        and access property lists, which enable creation of intermediate
        groups and proper handling of external links.
        """
        plists = {'lcpl': self._lcpl, 'lapl': self._lapl}

        if isinstance(obj, HLObject):
            h5o.link(obj.id, self.id, name, **plists)

        elif isinstance(obj, SoftLink):
            self.id.links.create_soft(name, obj.path, **plists)
        elif isinstance(obj, ExternalLink):
            self.id.links.create_external(name, obj.filename, obj.path, **plists)

        elif isinstance(obj, numpy.dtype):
            htype = h5t.py_create(obj)
            htype.commit(self.id, name, lcpl=self._lcpl)

        else:
            ds = self.create_dataset(None, data=obj, dtype=_guess_dtype(obj))
            h5o.link(ds.id, self.id, name, **plists)  

    def _set16(self, name, obj):
        """ HDF5 1.6 __setitem__.  PHIL should already be held. """
        if isinstance(obj, HLObject):
            self.id.link(h5i.get_name(obj.id), name, link_type=h5g.LINK_HARD)

        elif isinstance(obj, SoftLink):
            self.id.link(obj.path, name, link_type=h5g.LINK_SOFT)

        elif isinstance(obj, numpy.dtype):
            htype = h5t.py_create(obj)
            htype.commit(self.id, name)

        else:
            self.create_dataset(name, data=obj)

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
        register_thread()
        with phil:
            if config.API_18:
                self._set18(name, obj)
            else:
                self._set16(name, obj)

    def _get18(self, name):
        """ HDF5 1.8 __getitem__ 

        Works with string names.  Respects link access properties.
        """
        
        objinfo = h5o.get_info(self.id, name, lapl=self._lapl)

        cls = {h5o.TYPE_GROUP: Group, h5o.TYPE_DATASET: Dataset,
               h5o.TYPE_NAMED_DATATYPE: Datatype}.get(objinfo.type)
        if cls is None:
            raise TypeError("Unknown object type")

        oid = h5o.open(self.id, name, lapl=self._lapl)
        return cls(self, None, _rawid=oid)

    def _get16(self, name):
        """ HDF5 1.6 __getitem__ """
        objinfo = h5g.get_objinfo(self.id, name)
        
        cls = {h5g.DATASET: Dataset, h5g.GROUP: Group,
               h5g.TYPE: Datatype}.get(objinfo.type)
        if cls is None:
            raise TypeError("Unknown object type")
        
        return cls(self, name)
        
    def _getref(self, ref):
        """ Dereference and open (1.6 and 1.8) """
        cls = {h5g.DATASET: Dataset, h5g.GROUP: Group,
               h5g.TYPE: Datatype}.get(h5r.get_obj_type(ref, self.id))
        if cls is None:
            raise ValueError("Unrecognized object type")
    
        return cls(self, None, _rawid=h5r.dereference(ref, self.id))

    def __getitem__(self, name):
        """ Open an object attached to this group. 
        """
        register_thread()
        with phil:

            if isinstance(name, h5r.Reference):
                return self._getref(name)
            elif config.API_18:
                return self._get18(name)
            else:
                return self._get16(name)

    def __delitem__(self, name):
        """ Delete (unlink) an item from this group. """
        register_thread()
        self.id.unlink(name)

    def __len__(self):
        """ Number of members attached to this group """
        register_thread()
        return self.id.get_num_objs()

    def __contains__(self, name):
        """ Test if a member name exists """
        register_thread()
        return name in self.id

    def __iter__(self):
        """ Iterate over member names """
        register_thread()
        return self.id.__iter__()

    def create_group(self, name):
        """ Create and return a subgroup. Fails if the group already exists.
        """
        return Group(self, name, create=True)

    def require_group(self, name):
        """ Check if a group exists, and create it if not.  TypeError if an
        incompatible object exists.
        """
        with phil:
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

        with phil:
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
        register_thread()
        with phil:

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

        If the source is a Group object, all objects contained in that group
        will be copied recursively.

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
        register_thread()
        if not config.API_18:
            raise NotImplementedError("This feature is only available with HDF5 1.8.0 and later")

        with phil:

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
        register_thread()
        if not config.API_18:
            raise NotImplementedError("This feature is only available with HDF5 1.8.0 and later")
    
        with phil:
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
        register_thread()
        if not config.API_18:
            raise NotImplementedError("This feature is only available with HDF5 1.8.0 and later")

        with phil:
            def call_proxy(name):
                return func(name, self[name])
            return h5o.visit(self.id, call_proxy)

    def __repr__(self):
        if not self:
            return "<Closed HDF5 group>"
        namestr = '"%s"' % self.name if self.name is not None else "(anonymous)"
        return '<HDF5 group %s (%d members)>' % \
            (namestr, len(self))


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
                            Default is 64k.

        'family'
            Store the file on disk as a series of fixed-length chunks.  Useful
            if the file system doesn't allow large files.  Note: the filename
            you provide *must* contain a printf-style integer format code
            (e.g. %d"), which will be replaced by the file sequence number.
            Keywords:

            memb_size:  Maximum file size (default is 2**31-1).
    """

    _modes = weakref.WeakKeyDictionary()

    @property
    def filename(self):
        """File name on disk"""
        register_thread()
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

    @property
    def file(self):
        return self

    @property
    def mode(self):
        """Python mode used to open file"""
        register_thread()
        mode = self._modes.get(self)
        if mode is None and config.API_18:
            mode = {h5f.ACC_RDONLY: 'r', h5f.ACC_RDWR: 'r+'}.get(self.fid.get_intent())
        return mode

    @property
    def driver(self):
        """Low-level HDF5 file driver used to open file"""
        register_thread()
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
        register_thread()
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

            plist = self._generate_access_plist(driver, **kwds)
            self.fid = self._generate_fid(name, mode, plist)
            self._modes[self] = mode

        if config.API_18:
            gid = h5o.open(self.fid, '/', lapl=self._lapl)
        else:
            gid = h5g.open(self.fid, '/')
        Group.__init__(self, None, None, _rawid=gid)

    def _generate_access_plist(self, driver, **kwds):
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

    def _generate_fid(self, name, mode, plist):
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
        register_thread()
        with phil:
            while self.fid:
                self.fid.close()

    def flush(self):
        """ Tell the HDF5 library to flush its buffers.
        """
        register_thread()
        h5f.flush(self.fid)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        register_thread()
        with phil:
            if self.id._valid:
                self.close()
            
    def __repr__(self):
        register_thread()
        if not self:
            return "<Closed HDF5 file>"
        return '<HDF5 file "%s" (mode %s, %s)>' % \
            (os.path.basename(self.filename), self.mode,
             _extras.sizestring(self.fid.get_filesize()))


    def __hash__(self):
        register_thread()
        return hash(self.fid)
    def __eq__(self, other):
        # Python requires that objects which compare equal hash the same.
        # Therefore comparison to generic Group objects is impossible
        register_thread()
        if hasattr(other, 'fid'):
            return self.fid == other.fid
        return False

class _RegionProxy(object):

    def __init__(self, dset):
        self.id = dset.id

    def __getitem__(self, args):
        register_thread()
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

    # Internal properties

    def _g_shape(self):
        """Numpy-style shape tuple giving dataset dimensions"""
        register_thread()
        return self.id.shape

    def _s_shape(self, shape):
        self.resize(shape)

    shape = property(_g_shape, _s_shape)

    @_extras.cproperty('_dtype')
    def dtype(self):
        """Numpy dtype representing the datatype"""
        register_thread()
        return self.id.dtype

    @property
    def value(self):
        """  Deprecated alias for dataset[...] and dataset[()] """
        with phil:
            arr = self[...]
            #if arr.shape == ():
            #    return numpy.asscalar(arr)
            return arr

    @_extras.cproperty('__dcpl')
    def _dcpl(self):
        return self.id.get_create_plist()

    @_extras.cproperty('__filters')
    def _filters(self):
        return filters.get_filters(self._dcpl)

    @property
    def chunks(self):
        """Dataset chunks (or None)"""
        register_thread()
        dcpl = self._dcpl
        if dcpl.get_layout() == h5d.CHUNKED:
            return dcpl.get_chunk()
        return None

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
        register_thread()
        with phil:
            space = self.id.get_space()
            dims = space.get_simple_extent_dims(True)
            return tuple(x if x != h5s.UNLIMITED else None for x in dims)

    @property
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
        register_thread()
        with phil:
            if _rawid is not None:
                id = _rawid
            elif data is None and shape is None:
                if any((dtype,chunks,compression,shuffle,fletcher32)):
                    raise ValueError('You cannot specify keywords when opening a dataset.')
                id = h5d.open(group.id, name)
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

                id = h5d.create(group.id, name, type_id, space_id, plist)
                if data is not None:
                    id.write(h5s.ALL, h5s.ALL, data)
            HLObject.__init__(self, id)

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
        register_thread()
        with phil:

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
        register_thread()
        with phil:

            args = args if isinstance(args, tuple) else (args,)

            # Sort field indices from the rest of the args.
            names = tuple(x for x in args if isinstance(x, str))
            args = tuple(x for x in args if not isinstance(x, str))

            # Create NumPy datatype for read, using only the named fields
            # as specified by the user.
            basetype = self.id.dtype
            if len(names) == 0:
                new_dtype = basetype
            elif basetype.names is None:
                raise ValueError("Field names only allowed for compound types")
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
        register_thread()
        with phil:

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
        register_thread()
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
        register_thread()
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

    def __array__(self, dtype=None):
        with phil:
            arr = numpy.empty(self.shape, dtype=self.dtype if dtype is None else dtype)
            self.read_direct(arr)
            return arr

    def __repr__(self):
        if not self:
            return "<Closed HDF5 dataset>"
        namestr = '"%s"' % _extras.basename(self.name) if self.name is not None else "(anonymous)"
        return '<HDF5 dataset %s: shape %s, type "%s">' % \
            (namestr, self.shape, self.dtype.str)

class AttributeManager(_DictCompat):

    """ 
        Allows dictionary-style access to an HDF5 object's attributes.

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
        self._id = parent.id
        self._file = parent.file

    def __getitem__(self, name):
        """ Read the value of an attribute.
        """
        register_thread()
        with phil:
            attr = h5a.open(self._id, name)

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
        with phil:
            self.create(name, data=value, dtype=_guess_dtype(value))

    def __delitem__(self, name):
        """ Delete an attribute (which must already exist). """
        register_thread()
        h5a.delete(self._id, name)

    def create(self, name, data, shape=None, dtype=None):
        """ Create a new attribute, overwriting any existing attribute.

        name
            Name of the new attribute (required)
        data
            An array to initialize the attribute (required)
        shape
            Shape of the attribute.  Overrides data.shape if both are
            given.  The total number of points must be unchanged.
        dtype
            Data type of the attribute.  Overrides data.dtype if both
            are given.  Must be conversion-compatible with data.dtype.
        """
        register_thread()
        with phil:
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
                h5a.delete(self._id, name)

            attr = h5a.create(self._id, name, htype, space)
            if data is not None:
                attr.write(data)

    def modify(self, name, value):
        """ Change the value of an attribute while preserving its type.

        Differs from __setitem__ in that the type of an existing attribute
        is preserved.  Useful for interacting with externally generated files.

        If the attribute doesn't exist, it will be automatically created.
        """
        register_thread()
        with phil:
            if not name in self:
                self[name] = value
            else:
                value = numpy.asarray(value, order='C')

                attr = h5a.open(self._id, name)

                # Allow the case of () <-> (1,)
                if (value.shape != attr.shape) and not \
                   (numpy.product(value.shape)==1 and numpy.product(attr.shape)==1):
                    raise TypeError("Shape of data is incompatible with existing attribute")
                attr.write(value)

    def __len__(self):
        """ Number of attributes attached to the object. """
        # I expect we will not have more than 2**32 attributes
        register_thread()
        return h5a.get_num_attrs(self._id)

    def __iter__(self):
        """ Iterate over the names of attributes. """
        register_thread()
        with phil:
            attrlist = []
            def iter_cb(name, *args):
                attrlist.append(name)
            h5a.iterate(self._id, iter_cb)

            for name in attrlist:
                yield name

    def __contains__(self, name):
        """ Determine if an attribute exists, by name. """
        register_thread()
        return h5a.exists(self._id, name)

    def __repr__(self):
        if not self._id:
            return "<Attributes of closed HDF5 object>"
        return "<Attributes of HDF5 object at %s>" % id(self._id)

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
        register_thread()
        return self.id.dtype

    def __init__(self, grp, name, _rawid=None):
        """ Private constructor.
        """
        register_thread()
        with phil:
            id = _rawid if _rawid is not None else h5t.open(grp.id, name)
            HLObject.__init__(self, id)

    def __repr__(self):
        if not self.id:
            return "<Closed HDF5 named type>"
        namestr = '"%s"' % _extras.basename(self.name) if self.name is not None else "(anonymous)"
        return '<HDF5 named type %s (dtype %s)>' % \
            (namestr, self.dtype.str)

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

