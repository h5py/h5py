import posixpath as pp
import sys
import numpy

from h5py import h5s, h5t, h5r, h5d
from .base import HLObject
from . import filters
from . import selections as sel
from . import selections2 as sel2

def readtime_dtype(basetype, names):
    """ Make a NumPy dtype appropriate for reading """

    if basetype.kind == 'O':
        # Special-dtype fields break indexing
        basetype = numpy.dtype('O')

    if len(names) == 0:  # Not compound, or we want all fields
        return basetype

    if basetype.names is None:  # Names provided, but not compound
        raise ValueError("Field names only allowed for compound types")

    for name in names:  # Check all names are legal
        if not name in basetype.names:
            raise ValueError("Field %s does not appear in this type." % name)

    return numpy.dtype([(name, basetype.fields[name][0]) for name in names])

def make_new_dset(parent, shape=None, dtype=None, data=None,
                 chunks=None, compression=None, shuffle=None,
                    fletcher32=None, maxshape=None, compression_opts=None,
                  fillvalue=None):
    """ Return a new low-level dataset identifier

    Only creates anonymous datasets.
    """

    # Convert data to a C-contiguous ndarray
    if data is not None:
        import base
        data = numpy.asarray(data, order="C", dtype=base.guess_dtype(data))

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
    if any((compression, shuffle, fletcher32, maxshape)) and chunks is False:
        raise ValueError("Chunked format required for given storage options")

    # Legacy
    if compression is True:
        if compression_opts is None:
            compression_opts = 4
        compression = 'gzip'

    # Legacy
    if compression in range(10):
        if compression_opts is not None:
            raise TypeError("Conflict in compression options")
        compression_opts = compression
        compression = 'gzip'

    dcpl = filters.generate_dcpl(shape, dtype, chunks, compression, compression_opts,
                  shuffle, fletcher32, maxshape)

    if fillvalue is not None:
        fillvalue = numpy.array(fillvalue)
        dcpl.set_fill_value(fillvalue)

    if maxshape is not None:
        maxshape = tuple(m if m is not None else h5s.UNLIMITED for m in maxshape)
    sid = h5s.create_simple(shape, maxshape)
    tid = h5t.py_create(dtype, logical=1)

    dset_id = h5d.create(parent.id, None, tid, sid, dcpl=dcpl)

    if data is not None:
        dset_id.write(h5s.ALL, h5s.ALL, data)

    return dset_id

class _RegionProxy(object):

    def __init__(self, dset):
        self.id = dset.id

    def __getitem__(self, args):
        selection = sel.select(self.id.shape, args, dsid=self.id)
        return h5r.create(self.id, b'.', h5r.DATASET_REGION, selection._id)

class Dataset(HLObject):

    """
        Represents an HDF5 dataset
    """

    @property
    def dims(self):
        from . dims import DimensionManager
        return DimensionManager(self)

    @property
    def shape(self):
        """Numpy-style shape tuple giving dataset dimensions"""
        return self.id.shape
    @shape.setter
    def shape(self, shape):
        self.resize(shape)

    @property
    def size(self):
        """Numpy-style attribute giving the total dataset size"""
        return numpy.prod(self.shape)

    @property
    def dtype(self):
        """Numpy dtype representing the datatype"""
        return self.id.dtype

    @property
    def value(self):
        """  Alias for dataset[()] """
        DeprecationWarning("dataset.value has been deprecated. "
            "Use dataset[()] instead.")
        return self[()]

    @property
    def chunks(self):
        """Dataset chunks (or None)"""
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
        """Shape up to which this dataset can be resized.  Axes with value
        None have no resize limit. """
        space = self.id.get_space()
        dims = space.get_simple_extent_dims(True)
        return tuple(x if x != h5s.UNLIMITED else None for x in dims)

    @property
    def fillvalue(self):
        """Fill value for this dataset (0 by default)"""
        arr = numpy.ndarray((1,), dtype=self.dtype)
        dcpl = self._dcpl.get_fill_value(arr)
        return arr[0]

    @property
    def regionref(self):
        """Create a region reference.  The syntax is regionref[<slices>].
        For example, dset.regionref[...] creates a region reference in which
        the whole dataset is selected.
        """
        return _RegionProxy(self)

    def __init__(self, bind):
        """ Create a new Dataset object by binding to a low-level DatasetID.
        """
        if not isinstance(bind, h5d.DatasetID):
            raise ValueError("%s is not a DatasetID" % bind)
        HLObject.__init__(self, bind)

        self._dcpl = self.id.get_create_plist()
        self._filters = filters.get_filters(self._dcpl)

    def resize(self, size, axis=None):
        """ Resize the dataset, or the specified axis.

        The dataset must be stored in chunked format; it can be resized up to
        the "maximum shape" (keyword maxshape) specified at creation time.
        The rank of the dataset cannot be changed.

        "Size" should be a shape tuple, or if an axis is specified, an integer.

        BEWARE: This functions differently than the NumPy resize() method!
        The data is not "reshuffled" to fit in the new shape; each axis is
        grown or shrunk independently.  The coordinates of existing data are
        fixed.
        """
        if self.chunks is None:
            raise TypeError("Only chunked datasets can be resized")

        if axis is not None:
            if not (axis >=0 and axis < self.id.rank):
                raise ValueError("Invalid axis (0 to %s allowed)" % (self.id.rank-1))
            try:
                newlen = int(size)
            except TypeError:
                raise TypeError("Argument must be a single int if axis is specified")
            size = list(self.shape)
            size[axis] = newlen

        size = tuple(size)
        self.id.set_extent(size)
        #h5f.flush(self.id)  # THG recommends

    def __len__(self):
        """ The size of the first axis.  TypeError if scalar.

        Limited to 2**32 on 32-bit systems; Dataset.len() is preferred.
        """
        size = self.len()
        if size > sys.maxsize:
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
        """
        args = args if isinstance(args, tuple) else (args,)

        # Sort field indices from the rest of the args.
        names = tuple(x for x in args if isinstance(x, str))
        args = tuple(x for x in args if not isinstance(x, str))

        def strip_fields(basetype):
            """ Strip extra dtype information from special types """
            if basetype.kind == 'O':
                return numpy.dtype('O')
            if basetype.fields is not None:
                if basetype.kind in ('i','u'):
                    return basetype.fields['enum'][0]
                fields = []
                for name in basetype.names:
                    fff = basetype.fields[name]
                    if len(fff) == 3:
                        (subtype, offset, meta) = fff
                    else:
                        subtype, meta = fff
                        offset = 0
                    subtype = strip_fields(subtype)
                    fields.append((name, subtype))
                return numpy.dtype(fields)
            return basetype

        def readtime_dtype(basetype, names):
            """ Make a NumPy dtype appropriate for reading """

            basetype = strip_fields(basetype)

            if len(names) == 0:  # Not compound, or we want all fields
                return basetype

            if basetype.names is None:  # Names provided, but not compound
                raise ValueError("Field names only allowed for compound types")

            for name in names:  # Check all names are legal
                if not name in basetype.names:
                    raise ValueError("Field %s does not appear in this type." % name)

            return numpy.dtype([(name, basetype.fields[name][0]) for name in names])

        # This is necessary because in the case of array types, NumPy
        # discards the array information at the top level.
        new_dtype = readtime_dtype(self.id.dtype, names)
        mtype = h5t.py_create(new_dtype)

        # === Scalar dataspaces =================

        if self.shape == ():
            fspace = self.id.get_space()
            selection = sel2.select_read(fspace, args)
            arr = numpy.ndarray(selection.mshape, dtype=new_dtype)
            for mspace, fspace in selection:
                self.id.read(mspace, fspace, arr, mtype)
            if selection.mshape is None:
                return arr[()]
            return arr

        # === Everything else ===================

        # Perform the dataspace selection.
        selection = sel.select(self.shape, args, dsid=self.id)

        if selection.nselect == 0:
            return numpy.ndarray((0,), dtype=new_dtype)

        # Up-converting to (1,) so that numpy.ndarray correctly creates
        # np.void rows in case of multi-field dtype. (issue 135)
        single_element = selection.mshape == ()
        mshape = (1,) if single_element else selection.mshape
        arr = numpy.ndarray(mshape, new_dtype, order='C')

        # HDF5 has a bug where if the memory shape has a different rank
        # than the dataset, the read is very slow
        if len(mshape) < len(self.shape):
            # pad with ones
            mshape = (1,)*(len(self.shape)-len(mshape)) + mshape

        # Perfom the actual read
        mspace = h5s.create_simple(mshape)
        fspace = selection._id
        self.id.read(mspace, fspace, arr, mtype)

        # Patch up the output for NumPy
        if len(names) == 1:
            arr = arr[names[0]]     # Single-field recarray convention
        if arr.shape == ():
            arr = numpy.asscalar(arr)
        if single_element:
            arr = arr[0]
        return arr

    def __setitem__(self, args, val):
        """ Write to the HDF5 dataset from a Numpy array.

        NumPy's broadcasting rules are honored, for "simple" indexing
        (slices and integers).  For advanced indexing, the shapes must
        match.
        """
        args = args if isinstance(args, tuple) else (args,)

        # Sort field indices from the slicing
        names = tuple(x for x in args if isinstance(x, str))
        args = tuple(x for x in args if not isinstance(x, str))

        if len(names) != 0:
            raise TypeError("Field name selections are not allowed for write.")

        # Generally we try to avoid converting the arrays on the Python
        # side.  However, for compound literals this is unavoidable.
        if self.dtype.kind == "O" or (
          self.dtype.kind == 'V' and \
          (not isinstance(val, numpy.ndarray) or val.dtype.kind != 'V') ):
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
        Selections must be the output of numpy.s_[<args>].

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

        The source array must be C-contiguous.  Selections must be
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

    def __array__(self, dtype=None):
        """ Create a Numpy array containing the whole dataset.  DON'T THINK
        THIS MEANS DATASETS ARE INTERCHANGABLE WITH ARRAYS.  For one thing,
        you have to read the whole dataset everytime this method is called.
        """
        arr = numpy.empty(self.shape, dtype=self.dtype if dtype is None else dtype)
        self.read_direct(arr)
        return arr

    def __repr__(self):
        if not self:
            return "<Closed HDF5 dataset>"
        if self.name is None:
            namestr = '("anonymous")'
        else:
            name = pp.basename(pp.normpath(self.name))
            namestr = '"%s"' % (name if name != '' else '/')
        return '<HDF5 dataset %s: shape %s, type "%s">' % \
            (namestr, self.shape, self.dtype.str)



