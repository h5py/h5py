import posixpath as pp

import numpy

from h5py import h5g, h5i, h5o, h5r, h5t, h5l
from . import base
from .base import HLObject, DictCompat
from . import dataset
from . import datatype

class Group(HLObject, DictCompat):

    """ Represents an HDF5 group.
    """

    def __init__(self, bind):
        """ Create a new Group object by binding to a low-level GroupID.
        """
        if not isinstance(bind, h5g.GroupID):
            raise ValueError("%s is not a GroupID" % bind)
        HLObject.__init__(self, bind)

    def create_group(self, name):
        """ Create and return a new subgroup.

        Name may be absolute or relative.  Fails if the target name already
        exists.
        """
        name, lcpl = self._e(name, lcpl=True)
        gid = h5g.create(self.id, name, lcpl=lcpl)
        return Group(gid)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        """ Create a new HDF5 dataset

        name
            Name of the dataset (absolute or relative).  Provide None to make
            an anonymous dataset.
        shape
            Dataset shape.  Use "()" for scalar datasets.  Required if "data"
            isn't provided.
        dtype
            Numpy dtype or string.  If omitted, dtype('f') will be used.
            Required if "data" isn't provided; otherwise, overrides data
            array's dtype.
        data
            Provide data to initialize the dataset.  If used, you can omit
            shape and dtype arguments.

        Keyword-only arguments:

        chunks
            (Tuple) Chunk shape, or True to enable auto-chunking.
        maxshape
            (Tuple) Make the dataset resizable up to this shape.  Use None for
            axes you want to be unlimited.
        compression
            (String) Compression strategy.  Legal values are 'gzip', 'szip',
            'lzf'.  Can also use an integer in range(10) indicating gzip.
        compression_opts
            Compression settings.  This is an integer for gzip, 2-tuple for
            szip, etc.
        shuffle
            (T/F) Enable shuffle filter.
        fletcher32
            (T/F) Enable fletcher32 error detection.
        fillvalue
            (Scalar) Use this value for uninitialized parts of the dataset.
        """

        dsid = dataset.make_new_dset(self, shape, dtype, data, **kwds)
        dset = dataset.Dataset(dsid)
        if name is not None:
            self[name] = dset
        return dset

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        """ Open a dataset, creating it if it doesn't exist.

        If keyword "exact" is False (default), an existing dataset must have
        the same shape and a conversion-compatible dtype to be returned.  If
        True, the shape and dtype must match exactly.

        Other dataset keywords (see create_dataset) may be provided, but are
        only used if a new dataset is to be created.

        Raises TypeError if an incompatible object already exists, or if the
        shape or dtype don't match according to the above rules.
        """

        if not name in self:
            return self.create_dataset(name, *(shape, dtype), **kwds)

        dset = self[name]
        if not isinstance(dset, dataset.Dataset):
            raise TypeError("Incompatible object (%s) already exists" % dset.__class__.__name__)

        if not shape == dset.shape:
            raise TypeError("Shapes do not match (existing %s vs new %s)" % (dset.shape, shape))

        if exact:
            if not dtype == dset.dtype:
                raise TypeError("Datatypes do not exactly match (existing %s vs new %s)" % (dset.dtype, dtype))
        elif not numpy.can_cast(dtype, dset.dtype):
            raise TypeError("Datatypes cannot be safely cast (existing %s vs new %s)" % (dset.dtype, dtype))

        return dset

    def require_group(self, name):
        """ Return a group, creating it if it doesn't exist.

        TypeError is raised if something with that name already exists that
        isn't a group.
        """
        if not name in self:
            return self.create_group(name)
        grp = self[name]
        if not isinstance(grp, Group):
            raise TypeError("Incompatible object (%s) already exists" % grp.__class__.__name__)
        return grp

    def __getitem__(self, name):
        """ Open an object in the file """

        if isinstance(name, h5r.Reference):
            oid = h5r.dereference(name, self.id)
            if oid is None:
                raise ValueError("Invalid HDF5 object reference")
        else:
            oid = h5o.open(self.id, self._e(name), lapl=self._lapl)

        otype = h5i.get_type(oid)
        if otype == h5i.GROUP:
            return Group(oid)
        elif otype == h5i.DATASET:
            return dataset.Dataset(oid)
        elif otype == h5i.DATATYPE:
            return datatype.Datatype(oid)
        else:
            raise TypeError("Unknown object type")

    def get(self, name, default=None, getclass=False, getlink=False):
        """ Retrieve an item or other information.

        "name" given only:
            Return the item, or "default" if it doesn't exist

        "getclass" is True:
            Return the class of object (Group, Dataset, etc.), or "default"
            if nothing with that name exists

        "getlink" is True:
            Return HardLink, SoftLink or ExternalLink instances.  Return
            "default" if nothing with that name exists.

        "getlink" and "getclass" are True:
            Return HardLink, SoftLink and ExternalLink classes.  Return
            "default" if nothing with that name exists.

        Example:

        >>> cls = group.get('foo', getclass=True)
        >>> if cls == SoftLink:
        ...     print '"foo" is a soft link!'
        """
        if not name in self:
            return default

        if not (getclass or getlink):
            return self[name]

        elif getclass and not getlink:
            typecode = h5o.get_info(self.id, self._e(name)).type

            try:
                return {h5o.TYPE_GROUP: Group,
                        h5o.TYPE_DATASET: dataset.Dataset,
                        h5o.TYPE_NAMED_DATATYPE: datatype.Datatype}[typecode]
            except KeyError:
                raise TypeError("Unknown object type")

        elif getlink:
            typecode = self.id.links.get_info(self._e(name)).type

            if typecode == h5l.TYPE_SOFT:
                if getclass:
                    return SoftLink
                linkbytes = self.id.links.get_val(self._e(name))
                return SoftLink(self._d(linkbytes))
            elif typecode == h5l.TYPE_EXTERNAL:
                if getclass:
                    return ExternalLink
                filebytes, linkbytes = self.id.links.get_val(self._e(name))
                # TODO: I think this is wrong,
                # we should use filesystem decoding on the filename
                return ExternalLink(self._d(filebytes), self._d(linkbytes))
            elif typecode == h5l.TYPE_HARD:
                return HardLink if getclass else HardLink()
            else:
                raise TypeError("Unknown link type")

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
        name, lcpl = self._e(name, lcpl=True)

        if isinstance(obj, HLObject):
            h5o.link(obj.id, self.id, name, lcpl=lcpl, lapl=self._lapl)

        elif isinstance(obj, SoftLink):
            self.id.links.create_soft(name, self._e(obj.path),
                          lcpl=lcpl, lapl=self._lapl)

        elif isinstance(obj, ExternalLink):
            self.id.links.create_external(name, self._e(obj.filename),
                          self._e(obj.path), lcpl=lcpl, lapl=self._lapl)

        elif isinstance(obj, numpy.dtype):
            htype = h5t.py_create(obj)
            htype.commit(self.id, name, lcpl=lcpl)

        else:
            ds = self.create_dataset(None, data=obj, dtype=base.guess_dtype(obj))
            h5o.link(ds.id, self.id, name, lcpl=lcpl)

    def __delitem__(self, name):
        """ Delete (unlink) an item from this group. """
        self.id.unlink(self._e(name))

    def __len__(self):
        """ Number of members attached to this group """
        return self.id.get_num_objs()

    def __iter__(self):
        """ Iterate over member names """
        for x in self.id.__iter__():
            yield self._d(x)

    def __contains__(self, name):
        """ Test if a member name exists """
        return self._e(name) in self.id

    def copy(self, source, dest, name=None):
        """ Copy an object or group.

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
                # copy source into dest group: dest_name/source_name
                dest_path = pp.basename(h5i.get_name(source[source_path].id))

        elif isinstance(dest, HLObject):
            raise TypeError("Destination must be path or Group object")
        else:
            # Interpret destination as a path relative to this group
            dest_path = dest
            dest = self

        h5o.copy(source.id, self._e(source_path), dest.id, self._e(dest_path))

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
        def proxy(name):
            return func(self._d(name))
        return h5o.visit(self.id, proxy)

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
        def proxy(name):
            name = self._d(name)
            return func(name, self[name])
        return h5o.visit(self.id, proxy)

    def __repr__(self):
        if not self:
            return "<Closed HDF5 group>"
        namestr = '"%s"' % self.name if self.name is not None else "(anonymous)"
        return '<HDF5 group %s (%d members)>' % \
            (namestr, len(self))


class HardLink(object):

    """
        Represents a hard link in an HDF5 file.  Provided only so that
        Group.get works in a sensible way.  Has no other function.
    """

    pass

#TODO: implement equality testing for these
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
        self._filename = str(filename)
        self._path = str(path)

    def __repr__(self):
        return '<ExternalLink to "%s" in file "%s"' % (self.path, self.filename)


