.. _datasets:

========
Datasets
========

Datasets, like NumPy arrays, they are homogenous collections of data elements,
with an immutable datatype and (hyper)rectangular shape.  Unlike NumPy arrays,
they support a variety of transparent storage features such as compression,
error-detection, and chunked I/O.

They are represented in h5py by a thin proxy class which supports familiar
NumPy operations like slicing, along with a variety of descriptive attributes.

Datasets are created using either :meth:`Group.create_dataset` or
:meth:`Group.require_dataset`.  Existing datasets should be retrieved using
the group indexing syntax (``dset = group["name"]``).

NumPy compatibility
-------------------

Datasets implement the following parts of the NumPy-array user interface:

  - Slicing:  simple indexing and a subset of advanced indexing
  - **shape** attribute
  - **dtype** attribute

.. _dsetfeatures:

Special features
----------------

Unlike memory-resident NumPy arrays, HDF5 datasets support a number of optional
features.  These are enabled by the keywords provided to
:meth:`Group.create_dataset`.  Some of the more useful are:

Chunked storage
    HDF5 can store data in "chunks" indexed by B-trees, as well as in the
    traditional contiguous manner.  This can dramatically increase I/O
    performance for certain patterns of access; for example, reading every
    n-th element along the fastest-varying dimension.

Compression
    Transparent compression can substantially reduce
    the storage space needed for the dataset.  Beginning with h5py 1.1,
    three techniques are available, "gzip", "lzf" and "szip".  See the
    ``filters`` module for more information.

Error-Detection
    All versions of HDF5 include the *fletcher32* checksum filter, which enables
    read-time error detection for datasets.  If part of a dataset becomes
    corrupted, a read operation on that section will immediately fail with
    H5Error.

Resizing
    When using HDF5 1.8,
    datasets can be resized, up to a maximum value provided at creation time.
    You can specify this maximum size via the *maxshape* argument to
    :meth:`create_dataset <Group.create_dataset>` or
    :meth:`require_dataset <Group.require_dataset>`. Shape elements with the
    value ``None`` indicate unlimited dimensions.

    Later calls to :meth:`Dataset.resize` will modify the shape in-place::

        >>> dset = grp.create_dataset((10,10), '=f8', maxshape=(None, None))
        >>> dset.shape
        (10, 10)
        >>> dset.resize((20,20))
        >>> dset.shape
        (20, 20)

    Resizing an array with existing data works differently than in NumPy; if
    any axis shrinks, the data in the missing region is discarded.  Data does
    not "rearrange" itself as it does when resizing a NumPy array.

.. _slicing_access:

Slicing access
--------------

The best way to get at data is to use the traditional NumPy extended-slicing
syntax.   Slice specifications are translated directly to HDF5 *hyperslab*
selections, and are are a fast and efficient way to access data in the file.
The following slicing arguments are recognized:

    * Numbers: anything that can be converted to a Python long
    * Slice objects: please note negative values are not allowed
    * Field names, in the case of compound data
    * At most one ``Ellipsis`` (``...``) object

Here are a few examples (output omitted)

    >>> dset = f.create_dataset("MyDataset", data=numpy.ones((10,10,10),'=f8'))
    >>> dset[0,0,0]
    >>> dset[0,2:10,1:9:3]
    >>> dset[0,...]
    >>> dset[:,::2,5]

Simple array broadcasting is also supported:

    >>> dset[0]   # Equivalent to dset[0,...]

For compound data, you can specify multiple field names alongside the
numeric slices:

    >>> dset["FieldA"]
    >>> dset[0,:,4:5, "FieldA", "FieldB"]
    >>> dset[0, ..., "FieldC"]

Coordinate lists
----------------

For any axis, you can provide an explicit list of points you want; for a
dataset with shape (10, 10)::

    >>> dset.shape
    (10, 10)
    >>> result = dset[0, [1,3,8]]
    >>> result.shape
    (3,)
    >>> result = dset[1:6, [5,8,9]]
    >>> result.shape
    (5, 3)

The following restrictions exist:

* List selections may not be empty
* Selection coordinates must be given in increasing order
* Duplicate selections are ignored

.. _sparse_selection:

Sparse selection
----------------

Additional mechanisms exist for the case of scattered and/or sparse selection,
for which slab or row-based techniques may not be appropriate.

Boolean "mask" arrays can be used to specify a selection.  The result of
this operation is a 1-D array with elements arranged in the standard NumPy
(C-style) order:

    >>> arr = numpy.arange(100).reshape((10,10))
    >>> dset = f.create_dataset("MyDataset", data=arr)
    >>> result = dset[arr > 50]
    >>> result.shape
    (49,)

Advanced selection
------------------

The ``selections`` module contains additional classes which provide access to
HDF5 dataspace selection techniques, including point-based selection.  These 
are especially useful for read_direct and write_direct.

Length and iteration
--------------------

As with NumPy arrays, the ``len()`` of a dataset is the length of the first
axis.  Since Python's ``len`` is limited by the size of a C long, it's
recommended you use the syntax ``dataset.len()`` instead of ``len(dataset)``
on 32-bit platforms, if you expect the length of the first row to exceed 2**32.

Iterating over a dataset iterates over the first axis.  However, modifications
to the yielded data are not recorded in the file.  Resizing a dataset while
iterating has undefined results.

Reference
---------

.. class:: Dataset

    Represents an HDF5 dataset.  All properties are read-only.

    .. attribute:: name

        Full name of this dataset in the file (e.g. ``/grp/MyDataset``)

    .. attribute:: attrs

        Provides access to HDF5 attributes; see :ref:`attributes`.

    .. attribute:: file
        
        The ``File`` instance used to open this HDF5 file.

    .. attribute:: parent

        A group which contains this object, according to dirname(obj.name).

    .. attribute:: shape

        Numpy-style shape tuple with dataset dimensions

    .. attribute:: dtype

        Numpy dtype object representing the dataset type

    .. attribute:: value

        Special read-only property; for a regular dataset, it's equivalent to
        dset[:] (an ndarray with all points), but for a scalar dataset, it's
        a NumPy scalar instead of an 0-dimensional ndarray.

    .. attribute:: chunks

        Dataset chunk size, or None if chunked layout isn't used.

    .. attribute:: compression

        None or a string indicating the compression strategy;
        one of "gzip", "lzf", or "lzf".

    .. attribute:: compression_opts

        Setting for the compression filter

    .. attribute:: shuffle

        Is the shuffle filter being used? (T/F)

    .. attribute:: fletcher32

        Is the fletcher32 filter (error detection) being used? (T/F)

    .. attribute:: maxshape

        Maximum allowed size of the dataset, as specified when it was created.

    .. method:: __getitem__(*args) -> NumPy ndarray

        Read a slice from the dataset.  See :ref:`slicing_access`.

    .. method:: __setitem__(*args, val)

        Write to the dataset.  See :ref:`slicing_access`.

    .. method:: read_direct(dest, source_sel=None, dest_sel=None)

        Read directly from HDF5 into an existing NumPy array.  The "source_sel"
        and "dest_sel" arguments may be Selection instances (from the
        selections module) or the output of ``numpy.s_``.  Standard broadcasting
        is supported.

    .. method:: write_direct(source, source_sel=None, dest_sel=None)

        Write directly to HDF5 from a NumPy array.  The "source_sel"
        and "dest_sel" arguments may be Selection instances (from the
        selections module) or the output of ``numpy.s_``.  Standard broadcasting
        is supported.

    .. method:: resize(shape, axis=None)

        Change the size of the dataset to this new shape.  Must be compatible
        with the *maxshape* as specified when the dataset was created.  If
        the keyword *axis* is provided, the argument should be a single
        integer instead; that axis only will be modified.

        **Only available with HDF5 1.8**

    .. method:: __len__

        The length of the first axis in the dataset (TypeError if scalar).
        This **does not work** on 32-bit platforms, if the axis in question
        is larger than 2^32.  Use :meth:`len` instead.

    .. method:: len()

        The length of the first axis in the dataset (TypeError if scalar).
        Works on all platforms.

    .. method:: __iter__

        Iterate over rows (first axis) in the dataset.  TypeError if scalar.
