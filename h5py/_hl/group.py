# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements support for high-level access to HDF5 groups.
"""

from __future__ import absolute_import

import posixpath as pp
import six
import numpy
from copy import deepcopy as copy
from .compat import fsdecode
from .compat import fsencode
from .compat import fspath

from .. import h5g, h5i, h5o, h5r, h5t, h5l, h5p, h5s, h5d
from . import base
from .base import HLObject, MutableMappingHDF5, phil, with_phil
from . import dataset
from . import datatype
from types import EllipsisType
# from . import files
# # from .files import File


class Group(HLObject, MutableMappingHDF5):

    """ Represents an HDF5 group.
    """

    def __init__(self, bind):
        """ Create a new Group object by binding to a low-level GroupID.
        """
        with phil:
            if not isinstance(bind, h5g.GroupID):
                raise ValueError("%s is not a GroupID" % bind)
            HLObject.__init__(self, bind)

    def create_group(self, name):
        """ Create and return a new subgroup.

        Name may be absolute or relative.  Fails if the target name already
        exists.
        """
        with phil:
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
            (String or int) Compression strategy.  Legal values are 'gzip',
            'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        compression_opts
            Compression settings.  This is an integer for gzip, 2-tuple for
            szip, etc. If specifying a dynamically loaded compression filter
            number, this must be a tuple of values.
        scaleoffset
            (Integer) Enable scale/offset filter for (usually) lossy
            compression of integer or floating-point data. For integer
            data, the value of scaleoffset is the number of bits to
            retain (pass 0 to let HDF5 determine the minimum number of
            bits necessary for lossless compression). For floating point
            data, scaleoffset is the number of digits after the decimal
            place to retain; stored values thus have absolute error
            less than 0.5*10**(-scaleoffset).
        shuffle
            (T/F) Enable shuffle filter.
        fletcher32
            (T/F) Enable fletcher32 error detection. Not permitted in
            conjunction with the scale/offset filter.
        fillvalue
            (Scalar) Use this value for uninitialized parts of the dataset.
        track_times
            (T/F) Enable dataset creation timestamps.
        """
        with phil:
            dsid = dataset.make_new_dset(self, shape, dtype, data, **kwds)
            dset = dataset.Dataset(dsid)
            if name is not None:
                self[name] = dset
            return dset

    def create_virtual_dataset(self,VMlist, fillvalue=None):
        """
        Creates the virtual dataset from a list of virtual maps, any gaps are filled with a specified fill value.
        
        VMlist
            (List) A list of the the VirtualMaps between the source and target datasets. At least one is required.

        fillvalue
            (Scalar) Use this value for uninitialized parts of the dataset.
        """

        if not VMlist:
            raise ValueError("create_virtual_dataset requires at least one virtual map to construct output.")

        if not isinstance(VMlist,(tuple,list)):
            VMlist = [VMlist]
        with phil:
            dcpl = h5p.create(h5p.DATASET_CREATE)
            dcpl.set_fill_value(numpy.array([fillvalue]))
            sh = VMlist[0].target.shape
            virt_dspace = h5s.create_simple(sh, VMlist[0].target.maxshape) # create the virtual dataspace
            for VM in VMlist:
                virt_start_idx = tuple([ix.start for ix in VM.target.slice_list])
                virt_stride_index = tuple([ix.step for ix in VM.target.slice_list])
                if any(ix==h5s.UNLIMITED for ix in VM.target.maxshape):
                    count_idx = [1, ] * len(virt_stride_index)
                    unlimited_index = VM.target.maxshape.index(h5s.UNLIMITED)
                    count_idx[unlimited_index] = h5s.UNLIMITED
                    count_idx = tuple(count_idx)
                else:
                    count_idx = (1, ) * len(virt_stride_index)
                virt_dspace.select_hyperslab(start=virt_start_idx, 
                                             count=count_idx, 
                                             stride=virt_stride_index,
                                             block=VM.block_shape)
                dcpl.set_virtual(virt_dspace, VM.src.path, VM.src.key, VM.src_dspace)
            dset = h5d.create(self.id, name=VM.target.key, tid=h5t.py_create(VM.dtype,logical=1), space=virt_dspace, dcpl=dcpl)
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
        with phil:
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
        with phil:
            if not name in self:
                return self.create_group(name)
            grp = self[name]
            if not isinstance(grp, Group):
                raise TypeError("Incompatible object (%s) already exists" % grp.__class__.__name__)
            return grp

    @with_phil
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
        # pylint: disable=arguments-differ

        with phil:
            if not (getclass or getlink):
                try:
                    return self[name]
                except KeyError:
                    return default

            if not name in self:
                return default

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
                    return ExternalLink(fsdecode(filebytes), self._d(linkbytes))
                    
                elif typecode == h5l.TYPE_HARD:
                    return HardLink if getclass else HardLink()
                    
                else:
                    raise TypeError("Unknown link type")

    @with_phil
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
            self.id.links.create_external(name, fsencode(obj.filename),
                          self._e(obj.path), lcpl=lcpl, lapl=self._lapl)

        elif isinstance(obj, numpy.dtype):
            htype = h5t.py_create(obj, logical=True)
            htype.commit(self.id, name, lcpl=lcpl)

        else:
            ds = self.create_dataset(None, data=obj, dtype=base.guess_dtype(obj))
            h5o.link(ds.id, self.id, name, lcpl=lcpl)

    @with_phil
    def __delitem__(self, name):
        """ Delete (unlink) an item from this group. """
        self.id.unlink(self._e(name))

    @with_phil
    def __len__(self):
        """ Number of members attached to this group """
        return self.id.get_num_objs()

    @with_phil
    def __iter__(self):
        """ Iterate over member names """
        for x in self.id.__iter__():
            yield self._d(x)

    @with_phil
    def __contains__(self, name):
        """ Test if a member name exists """
        return self._e(name) in self.id

    def copy(self, source, dest, name=None,
             shallow=False, expand_soft=False, expand_external=False,
             expand_refs=False, without_attrs=False):
        """Copy an object or group.

        The source can be a path, Group, Dataset, or Datatype object.  The
        destination can be either a path or a Group object.  The source and
        destinations need not be in the same file.

        If the source is a Group object, all objects contained in that group
        will be copied recursively.

        When the destination is a Group object, by default the target will
        be created in that group with its current name (basename of obj.name).
        You can override that by setting "name" to a string.

        There are various options which all default to "False":

         - shallow: copy only immediate members of a group.

         - expand_soft: expand soft links into new objects.

         - expand_external: expand external links into new objects.

         - expand_refs: copy objects that are pointed to by references.

         - without_attrs: copy object without copying attributes.

       Example:

        >>> f = File('myfile.hdf5')
        >>> f.listnames()
        ['MyGroup']
        >>> f.copy('MyGroup', 'MyCopy')
        >>> f.listnames()
        ['MyGroup', 'MyCopy']

        """
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
                    # copy source into dest group: dest_name/source_name
                    dest_path = pp.basename(h5i.get_name(source[source_path].id))

            elif isinstance(dest, HLObject):
                raise TypeError("Destination must be path or Group object")
            else:
                # Interpret destination as a path relative to this group
                dest_path = dest
                dest = self

            flags = 0
            if shallow:
                flags |= h5o.COPY_SHALLOW_HIERARCHY_FLAG
            if expand_soft:
                flags |= h5o.COPY_EXPAND_SOFT_LINK_FLAG
            if expand_external:
                flags |= h5o.COPY_EXPAND_EXT_LINK_FLAG
            if expand_refs:
                flags |= h5o.COPY_EXPAND_REFERENCE_FLAG
            if without_attrs:
                flags |= h5o.COPY_WITHOUT_ATTR_FLAG
            if flags:
                copypl = h5p.create(h5p.OBJECT_COPY)
                copypl.set_copy_object(flags)
            else:
                copypl = None

            h5o.copy(source.id, self._e(source_path), dest.id, self._e(dest_path),
                     copypl, base.dlcpl)

    def move(self, source, dest):
        """ Move a link to a new location in the file.

        If "source" is a hard link, this effectively renames the object.  If
        "source" is a soft or external link, the link itself is moved, with its
        value unmodified.
        """
        with phil:
            if source == dest:
                return
            self.id.links.move(self._e(source), self.id, self._e(dest),
                               lapl=self._lapl, lcpl=self._lcpl)

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
        with phil:
            def proxy(name):
                """ Call the function with the text name, not bytes """
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
        with phil:
            def proxy(name):
                """ Use the text name of the object, not bytes """
                name = self._d(name)
                return func(name, self[name])
            return h5o.visit(self.id, proxy)

    @with_phil
    def __repr__(self):
        if not self:
            r = six.u("<Closed HDF5 group>")
        else:
            namestr = (
                six.u('"%s"') % self.name
            ) if self.name is not None else six.u("(anonymous)")
            r = six.u('<HDF5 group %s (%d members)>') % (namestr, len(self))

        if six.PY2:
            return r.encode('utf8')
        return r


class HardLink(object):

    """
        Represents a hard link in an HDF5 file.  Provided only so that
        Group.get works in a sensible way.  Has no other function.
    """

    pass


class SoftLink(object):

    """
        Represents a symbolic ("soft") link in an HDF5 file.  The path
        may be absolute or relative.  No checking is performed to ensure
        that the target actually exists.
    """

    @property
    def path(self):
        """ Soft link value.  Not guaranteed to be a valid path. """
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
        """ Soft link path, i.e. the part inside the HDF5 file. """
        return self._path

    @property
    def filename(self):
        """ Path to the external HDF5 file in the filesystem. """
        return self._filename

    def __init__(self, filename, path):
        self._filename = fspath(filename)
        self._path = str(path)

    def __repr__(self):
        return '<ExternalLink to "%s" in file "%s"' % (self.path, self.filename)


class DatasetContainer(object):
    def __init__(self, path, key,shape, dtype=None, maxshape=None):
        """
        This is an object that looks like a dataset, but it not. It allows the user to specify the maps based on lazy indexing, 
        which is natural, but without needing to load the data. 
                
        path
            This is the full path to the file on disk.
        
        key
            This is the key to the entry inside the hdf5 file.
        
        shape
            The shape of the data. We specify this by hand because it is a lot faster than getting it from the source file.
        
        dtype
            The data type. For the source we specify this because it is faster than getting from the file. For the target,
            we can specify it to be different to the source.
        """
        self.path = path
        self.key = key
        self.shape = shape
        self.slice_list = [slice(0,ix,1) for ix in self.shape] # if we don't slice, we want the whole array
        if maxshape is None:
            self.maxshape=shape
        else:
            self.maxshape = tuple([h5s.UNLIMITED if ix is None else ix for ix in maxshape])


    def _parse_slicing(self, key):
        """
        parses the __get_item__ key to get useful slicing information
        """
        tmp = copy(self)
        rank = len(self.shape)
        if (rank-len(key))<0:
            raise IndexError('Index rank is greater than dataset rank')
        if isinstance(key[0], tuple): # sometimes this is needed. odd
            key = key[0]
        key = list(key)
        key = [slice(ix, ix + 1, 1) if isinstance(ix, (int, float)) else ix for ix in key]
        
        # now let's parse ellipsis
        ellipsis_test = [ix==Ellipsis for ix in key]
        if sum(ellipsis_test)>1:
            raise ValueError("Only use of one Ellipsis(...) supported.")
        if not any(ellipsis_test):
            tmp.slice_list[:len(key)] = key
        elif any(ellipsis_test) and (len(key) is not 1):
            ellipsis_idx = ellipsis_test.index(True)
            ellipsis_idx_back = ellipsis_test[::-1].index(True)
            tmp.slice_list[0:ellipsis_idx] = key[0:ellipsis_idx]
            if ellipsis_idx_back>=ellipsis_idx: # edge case
                tmp.slice_list[(-ellipsis_idx_back):] = key[(-ellipsis_idx_back):]

        new_shape = []
        for ix, sl in enumerate(tmp.slice_list):
            step = 1 if sl.step is None else sl.step
            if step > 0:
                start = 0 if sl.start is None else sl.start# parse for Nones
                stop = self.shape[ix] if sl.stop is None else sl.stop
                start = self.shape[ix]+start if start<0 else start
                stop = self.shape[ix]+stop if stop<0 else stop
                if start < stop:
                    new_shape.append((stop - start + step - 1)/step)
                else:
                    new_shape.append(0)

            elif step < 0:
                stop = 0 if sl.stop is None else sl.stop# parse for Nones
                start = self.shape[ix] if sl.start is None else sl.start
                
                start = self.shape[ix]+start if start<0 else start
                stop = self.shape[ix]+stop if stop<0 else stop

                if start > stop: # this gets the same behaviour as numpy array
                    new_shape.append((start - stop - step - 1)/-step)
                else:
                    new_shape.append(0)
            elif step == 0:
                raise IndexError("A step of 0 is not valid")
            tmp.slice_list[ix] = slice(start, stop, step)
        tmp.shape = tuple(new_shape)
        return tmp

class VirtualSource(DatasetContainer):
    """
    A container for the source information. This is similar to a virtual target, but the shape information changes with slicing.
    This does not happen with VirtualTarget since it is the source that ultimately set's the block shape.
    """
    def __getitem__(self, *key):
        tmp =self._parse_slicing(key)
        return tmp

class VirtualTarget(DatasetContainer):
    """
    A container for the target information. This is similar to a virtual source, but the shape information does not change with slicing.
    This does not happen with VirtualSource since it is the source that ultimately set's the block shape so it must change on slicing.
    """
    def __getitem__(self, *key):
        tmp =self._parse_slicing(key)
        tmp.shape = self.shape
        return tmp


class VirtualMap(object):
    def __init__(self, virtual_source, virtual_target, dtype):
        """
        The idea of this class is to specify the mapping between the source and target files.
        Since data type casting is supported by VDS, we include this here.
            virtual_source
                A DatasetContainer object containing all the useful information about the source file for this map.
            
            virtual_target
                A DatasetContainer object containing all the useful information about the source file for this map.

            dtype
                The type of the final output dataset.
        
        """
        self.src = virtual_source[...]
        self.dtype = dtype
        self.target = virtual_target[...]
        self.block_shape = None
        # if the rank of the two datasets is not the same, pad with singletons. This isn't necessarily the best way to do this!
        rank_def = len(self.target.shape) - len(self.src.shape)
        if rank_def > 0:
            if len(self.src.shape)==1:
                pass
            else:
                self.block_shape = (1,)*rank_def + self.src.shape
        elif rank_def < 0:
            # This might be pathological.
            if len(self.target.shape)==1:
                pass
            else:
                self.block_shape = (1,)*rank_def + self.target.shape
        else:
            self.block_shape = self.src.shape

        self.src_dspace = h5s.create_simple(self.src.shape, self.src.maxshape)

        start_idx = tuple([ix.start for ix in self.src.slice_list])
        stride_idx = tuple([ix.step for ix in self.src.slice_list])

        if any(ix==h5s.UNLIMITED for ix in self.src.maxshape):
            count_idx = [1, ] * len(stride_idx)
            unlimited_index = self.src.maxshape.index(h5s.UNLIMITED)
            count_idx[unlimited_index] = h5s.UNLIMITED
            count_idx = tuple(count_idx)
            bs = list(self.block_shape)
            bs[unlimited_index] = 1
            self.block_shape = tuple(bs)
        else:
            count_idx = (1, ) * len(stride_idx)
        self.src_dspace.select_hyperslab(start=start_idx, count=count_idx, stride=stride_idx, block=self.block_shape)
