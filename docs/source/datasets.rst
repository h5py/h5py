****************
Datasets in HDF5
****************

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

There are two ways to create a dataset, with nearly identical syntax.  The
recommended procedure is to use a method on the Group object in which the
dataset will be stored:

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


Slicing and data access
=======================

A subset of the NumPy extended slicing is supported.  Slice specifications are
translated directly to HDF5 *hyperslab* selections, and are are a fast and
efficient way to access data in the file.










