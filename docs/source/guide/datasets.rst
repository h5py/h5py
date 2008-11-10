.. _Datasets:

**************
Using Datasets
**************

Datasets are where most of the information in an HDF5 file resides.  Like
NumPy arrays, they are homogenous collections of data elements, with an
immutable datatype and (hyper)rectangular shape.  Unlike NumPy arrays, they
support a variety of transparent storage features such as compression,
error-detection, and chunked I/O.

Metadata can be associated with an HDF5 dataset in the form of an "attribute".
It's recommended that you use this scheme for any small bits of information
you want to associate with the dataset.  For example, a descriptive title,
digitizer settings, or data collection time are appropriate things to store
as HDF5 attributes.


Opening an existing dataset
===========================

Since datasets reside in groups, the best way to retrive a dataset is by
indexing the group directly:

    >>> dset = grp["Dataset Name"]

You can also open a dataset by passing the group and name directly to the
constructor:

    >>> dset = Dataset(grp, "Dataset Name")

No options can be specified when opening a dataset, as almost all properties
of datasets are immutable.


Creating a dataset
==================

There are two ways to explicitly create a dataset, with nearly identical
syntax.  The recommended procedure is to use a method on the Group object in
which the dataset will be stored:

    >>> dset = grp.create_dataset("Dataset Name", ...options...)

Or you can call the Dataset constructor.  When providing more than just the
group and name, the constructor will try to create a new dataset:

    >>> dset = Dataset(grp, "Dataset name", ...options...)

Bear in mind that if an object of the same name already exists in the group,
you will have to manually unlink it first:

    >>> "Dataset Name" in grp
    True
    >>> del grp["Dataset name"]
    >>> dset = grp.create_dataset("Dataset Name", ...options...)

Logically, there are two ways to specify a dataset; you can tell HDF5 its
shape and datatype explicitly, or you can provide an existing ndarray from
which the shape, dtype and contents will be determined.  The following options
are used to communicate this information.


Arguments and options
---------------------

All options below can be given to either the Dataset constructor or the
Group method create_dataset.  They are listed in the order the arguments are
taken for both methods.  Default values are in *italics*.

*   **shape** = *None* or tuple(<dimensions>)

    A Numpy-style shape tuple giving the dataset dimensions.  Required if
    option **data** isn't provided.

*   **dtype** = *None* or NumPy dtype

    A NumPy dtype, or anything from which a dtype can be determined.
    This sets the datatype.  If this is omitted, the dataset will
    consist of single-precision floats, in native byte order ("=f4").

*   **data** = *None* or ndarray

    A NumPy array.  The dataset shape and dtype will be determined from
    this array, and the dataset will be initialized to its contents.
    Required if option **shape** isn't provided.

*   **chunks** = *None* or tuple(<chunk dimensions>)

    Manually set the HDF5 chunk size.

    When using any of the following options like compression or error-
    detection, the dataset is stored in chunked format, as small atomic
    pieces of data on which the filters operate.  These chunks are then
    indexed by B-trees.  Ordinarily h5py will guess a chunk value.  If you
    know what you're doing, you can override that value here.

*   **compression** = *None* or int(0-9)

    Enable the use of GZIP compression, at the given integer level.  The
    dataset will be stored in chunked format.

*   **shuffle** = True / *False*

    Enable the shuffle filter, possibly increasing the GZIP compression
    ratio.  The dataset will be stored in chunked format.

*   **fletcher32** = True / *False*

    Enable Fletcher32 error-detection.  The dataset will be stored in
    chunked format.

*   **maxshape** = *None* or tuple(<dimensions>)

    If provided, the dataset will be stored in a chunked and extendible fashion.
    The value provided should be a tuple of integers indicating the maximum
    size of each axis.  You can provide a value of "None" for any axis to
    indicate that the maximum size of that dimension is unlimited.

Automatic creation
------------------

If you've already got a NumPy array you want to store, you can let h5py guess
these options for you.  Simply assign the array to a Group entry:

    >>> arr = numpy.ones((100,100), dtype='=f8')
    >>> my_group["MyDataset"] = arr

The object you provide doesn't even have to be an ndarray; if it isn't, h5py
will create an intermediate NumPy representation before storing it.
The resulting dataset is stored contiguously, with no compression or chunking.

.. note::
    Arrays are auto-created using the NumPy ``asarray`` function.  This means
    that if you try to create a dataset from a string, you'll get a *scalar*
    dataset containing the string itself!  To get a char array, pass in
    something like ``numpy.fromstring(mystring, '|S1')`` instead.


Data Access and Slicing
=======================

A subset of the NumPy indexing techniques is supported, including the
traditional extended-slice syntax, named-field access, and boolean arrays.
Discrete coordinate selection are also supported via an special indexer class.

Properties
----------

Like Numpy arrays, Dataset objects have attributes named "shape" and "dtype":

    >>> dset.dtype
    dtype('complex64')
    >>> dset.shape
    (4L, 5L)

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

Advanced indexing
-----------------

Boolean "mask" arrays can also be used to specify a selection.  The result of
this operation is a 1-D array with elements arranged in the standard NumPy
(C-style) order:

    >>> arr = numpy.random.random((10,10))
    >>> dset = f.create_dataset("MyDataset", data=arr)
    >>> result = dset[arr > 0.5]

If you have a set of discrete points you want to access, you may not want to go
through the overhead of creating a boolean mask.  This is especially the case
for large datasets, where even a byte-valued mask may not fit in memory.  You
can pass a list of points to the dataset selector via a custom "CoordsList"
instance:

    >>> mycoords = [ (0,0), (3,4), (7,8), (3,5), (4,5) ]
    >>> coords_list = CoordsList(mycoords)
    >>> result = dset[coords_list]

Like boolean-array indexing, the result is a 1-D array.  The order in which
points are selected is preserved.

.. note::
    These two techniques rely on an HDF5 construct which explicitly enumerates the
    points to be selected.  It's very flexible but most appropriate for 
    reasonably-sized (or sparse) selections.  The coordinate list takes at
    least 8*<rank> bytes per point, and may need to be internally copied.  For
    example, it takes 40MB to express a 1-million point selection on a rank-3
    array.  Be careful, especially with boolean masks.

Value attribute and scalar datasets
-----------------------------------

HDF5 allows you to store "scalar" datasets.  These have the shape "()".  You
can use the syntax ``dset[...]`` to recover the value as an 0-dimensional
array.  Also, the special attribute ``value`` will return a scalar for an 0-dim
array, and a full n-dimensional array for all other cases:

    >>> f["ArrayDS"] = numpy.ones((2,2))
    >>> f["ScalarDS"] = 1.0
    >>> f["ArrayDS"].value
    array([[ 1.,  1.],
           [ 1.,  1.]])
    >>> f["ScalarDS"].value
    1.0

Extending Datasets
------------------

If the dataset is created with the *maxshape* option set, you can later expand
its size.  Simply call the *extend* method:

    >>> dset = f.create_dataset("MyDataset", (5,5), maxshape=(None,None))
    >>> dset.shape
    (5, 5)
    >>> dset.extend((15,20))
    >>> dset.shape
    (15, 20)

More on Datatypes
=================

Storing compound data
---------------------

You can store "compound" data (struct-like, using named fields) using the Numpy
facility for compound data types.  For example, suppose we have data that takes
the form of (temperature, voltage) pairs::

    >>> import numpy
    >>> mydtype = numpy.dtype([('temp','=f4'),('voltage','=f8')])
    >>> dset = f.create_dataset("MyDataset", (20,30), mydtype)
    >>> dset
    Dataset "MyDataset": (20L, 30L) dtype([('temp', '<f4'), ('voltage', '<f8')])
    
These types may contain any supported type, and be arbitrarily nested.

.. _supported:

Supported types
-----------------

The HDF5 type system is mostly a superset of its NumPy equivalent.  The
following are the NumPy types currently supported by the interface:

    ========    ==========  ==========  ===============================
    Datatype    NumPy kind  HDF5 class  Notes
    ========    ==========  ==========  ===============================
    Integer     i, u        INTEGER
    Float       f           FLOAT
    Complex     c           COMPOUND    Stored as an HDF5 struct
    Array       V           ARRAY       NumPy array with "subdtype"
    Opaque      V           OPAQUE      Stored as HDF5 fixed-length opaque
    Compound    V           COMPOUND    May be arbitarily nested
    String      S           STRING      Stored as HDF5 fixed-length C-style strings
    ========    ==========  ==========  ===============================

Byte order is always preserved.  The following additional features are known
not to be supported:

    * Read/write HDF5 variable-length (VLEN) data

      No obvious way exists to handle variable-length data in NumPy.

    * NumPy object types (dtype "O")

      This could potentially be solved by pickling, but requires low-level
      VLEN infrastructure.

    * HDF5 enums

      There's no NumPy dtype support for enums.  Enum data is read as plain
      integer data.  However, the low-level conversion routine
      ``h5t.py_create`` can create an HDF5 enum from a integer dtype and a
      dictionary of names.
    
    * HDF5 "time" datatype

      This datatype is deprecated, and has no close NumPy equivalent.

    
     







