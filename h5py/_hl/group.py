import numpy
import base
from base import HLObject, DictCompat
from h5py import h5g, h5i, h5o, h5r, h5t, h5l
import dataset
import datatype

class Group(HLObject, DictCompat):

    """ Represents an HDF5 group.
    """

    def __init__(self, parent_object, name, create=False, bind=None):
        """ Don't manually create Group objects.  Use File.create_group to
        make new groups, and indexing syntax (file['name']) to open them.

        This constructor is provided for backwards compatibility only.
        """
        if bind is None:
            # Old constructor used to do things
            if create:
                bind = h5g.create(self.id, name, lcpl=self._lcpl)
            else:
                bind = get(parent_object, name)
                if not isinstance(bind, h5g.GroupID):
                    raise TypeError("%s is not an HDF5 group" % name)

        HLObject.__init__(self, bind)
        
    def create_group(self, name):
        """ Create and return a new subgroup.

        Name may be absolute or relative.  Fails if the target name already
        exists.
        """
        gid = h5g.create(self.id, name, lcpl=self._lcpl)
        return Group(None, None, bind=gid)

    def create_dataset(self, name, shape=None, dtype=None, data=None,
                 chunks=None, compression=None, shuffle=None,
                    fletcher32=None, maxshape=None, compression_opts=None):
        dsid = dataset.make_new_dset(self, name, shape, dtype, data, chunks,
                compression, shuffle, fletcher32, maxshape, compression_opts)
        return dataset.Dataset(None, None, bind=dsid)

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        #TODO
        pass

    def require_group(self, name):
        """ Return group, creating it if it doesn't exist.  TypeError raised
        if something with that name already exists that isn't a group.
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
            oid = h5o.open(self.id, name, lapl=self._lapl)

        otype = h5i.get_type(oid)
        if otype == h5i.GROUP:
            return Group(None, None, bind=oid)
        elif otype == h5i.DATASET:
            return dataset.Dataset(None, None, bind=oid)
        elif otype == h5i.DATATYPE:
            return datatype.Datatype(None, None, bind=oid)
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
            typecode = h5o.get_info(self.id, name).type

            try:
                return {h5o.TYPE_GROUP: Group,
                        h5o.TYPE_DATASET: dataset.Dataset,
                        h5o.TYPE_NAMED_DATATYPE: datatype.Datatype}[typecode]
            except KeyError:
                raise TypeError("Unknown object type")

        elif getlink:
            typecode = self.id.links.get_info(name).type

            if typecode == h5l.TYPE_SOFT:
                return SoftLink if getclass else SoftLink(self.id.links.get_val(name))
            elif typecode == h5l.TYPE_EXTERNAL:
                return ExternalLink if getclass else ExternalLink(*self.id.links.get_val(name))
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
        if isinstance(obj, HLObject):
            h5o.link(obj.id, self.id, name, lcpl=self._lcpl, lapl=self._lapl)

        elif isinstance(obj, SoftLink):
            self.id.links.create_soft(name, obj.path, lcpl=self._lcpl, lapl=self._lapl)

        elif isinstance(obj, ExternalLink):
            self.id.links.create_external(name, obj.filename, obj.path, lcpl=self._lcpl, lapl=self._lapl)

        elif isinstance(obj, numpy.dtype):
            htype = h5t.py_create(obj)
            htype.commit(self.id, name, lcpl=self._lcpl)

        else:
            ds = self.create_dataset(None, data=obj, dtype=base.guess_dtype(obj))
            h5o.link(ds.id, self.id, name, lcpl=self._lcpl)

    def __delitem__(self, name):
        """ Delete (unlink) an item from this group. """
        self.id.unlink(name)

    def __len__(self):
        """ Number of members attached to this group """
        return self.id.get_num_objs()

    def __iter__(self):
        """ Iterate over member names """
        return self.id.__iter__()

    def __contains__(self, name):
        """ Test if a member name exists """
        return name in self.id

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
        def call_proxy(name):
            return func(name, self[name])
        return h5o.visit(self.id, call_proxy)

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


