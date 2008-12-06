
.. _h5pyreference:

*************
Documentation
*************

.. module:: h5py.highlevel

The high-level interface is the most convenient method to talk to HDF5.  There
are three main abstractions: files, groups, and datasets. Each is documented
separately below.

You may want to read the :ref:`quick start guide <quick>` to get a general
overview.

Everything useful in this module is automatically exported to the `h5py`
package namespace; you can do::

    >>> from h5py import *  # or from h5py import File, etc.

General information about h5py and HDF5
=======================================

Paths in HDF5
-------------

HDF5 files are organized like a filesystem.  :class:`Group` objects work like
directories; they can contain other groups, and :class:`Dataset` objects.  Like
a POSIX filesystem, objects are specified by ``/``-separated names, with the
root group ``/`` (represented by the :class:`File` class) at the base.

Wherever a name or path is called for, it may be relative or absolute.
Constructs like ``..`` (parent group) are allowed.


Exceptions
----------

Standard Python exceptions like TypeError and ValueError are raised in
response to inappropriate arguments.  When an error is encountered by the
HDF5 library itself, h5py.H5Error (or more commonly, a subclass) is raised.
For example::

    >>> try:
    >>>     myfile = h5py.File(some_name)
    >>> except TypeError:
    >>>     print "Argument some_name isn't a string!"
    >>> except H5Error:
    >>>     print "Problem opening the file %s" % some_name

In practice, all H5Error exceptions contain a useful stacktrace generated
by the library itself, including a good description of where and why the error
occurred.

Library configuration
---------------------

A few library options are available to change the behavior of the library.
You can get a reference to the global library configuration object via the
function ``h5py.get_config()``.  This object supports the following attributes:

    **complex_names**
        Set to a 2-tuple of strings (real, imag) to control how complex numbers
        are saved.  The default is ('r','i').

Threading
---------

H5py is now always thread-safe.  As HDF5 does not support thread-level
concurrency (and as it is not necessarily thread-safe), only one thread
at a time can acquire the lock which manages access to the library.

File compatibility
------------------

HDF5 in general (and h5py in particular) tries to be as backwards-compatible
as possible when writing new files.  However, certain combinations of dataset
filters may cause issues when attempting to read files created with HDF5 1.8
from an installation using HDF5 1.6.  It's generally best to use the same
version of HDF5 for all your applications.

Metadata
--------

Every object in HDF5 supports metadata in the form of "attributes", which are
small, named bits of data.  :class:`Group`, :class:`Dataset` and even
:class:`File` objects each carry a dictionary-like object which exposes this
behavior, named ``<obj>.attrs``.  This is the correct way to store metadata
in HDF5 files.

--------------------------------------------------------

File Objects
============

To open an HDF5 file, just instantiate the File object directly::

    >>> from h5py import File  # or import *
    >>> file_obj = File('myfile.hdf5','r')

Valid modes (like Python's file() modes) are:

    ===  ================================================
     r   Readonly, file must exist
     r+  Read/write, file must exist
     w   Create file, truncate if exists
     w-  Create file, fail if exists
     a   Read/write if exists, create otherwise (default)
    ===  ================================================

Like Python files, you should close the file when done::

    >>> file_obj.close()

File objects can also be used as "context managers" along with the new Python
``with`` statement.  When used in a ``with`` block, they will be closed at
the end of the block, even if an exception has been raised::

    >>> with File('myfile.hdf5', 'r') as file_obj:
    ...    # do stuff with file_obj
    ...
    >>> # file_obj is guaranteed closed at end of block

.. note::

    In addition to the methods and properties listed below, File objects also
    have all the methods and properties of :class:`Group` objects.  In this
    case the group in question is the HDF5 *root group* (``/``).

Reference
---------

.. class:: File

    Represents an HDF5 file on disk.

    .. attribute:: name

        HDF5 filename

    .. attribute:: mode

        Mode used to open file

    .. method:: __init__(name, mode='a')
        
        Open or create an HDF5 file.

    .. method:: close()

        Close the file.  Like Python files, you should call this when
        finished to be sure your data is saved.

    .. method:: flush()

        Ask the HDF5 library to flush its buffers for this file.


Groups
======

Groups are the container mechanism by which HDF5 files are organized.  From
a Python perspective, they operate somewhat like dictionaries.  In this case
the "keys" are the names of group entries, and the "values" are the entries
themselves (:class:`Group` and :class:`Dataset`) objects.  Objects are
retrieved from the file using the standard indexing notation::

    >>> file_obj = File('myfile.hdf5')
    >>> subgroup = file_obj['/subgroup']
    >>> dset = subgroup['MyDataset']  # full name /subgroup/Mydataset

Objects can be deleted from the file using the standard syntax::

    >>> del subgroup["MyDataset"]

However, new groups and datasets should generally be created using method calls
like :meth:`create_group <Group.create_group>` or
:meth:`create_dataset <Group.create_dataset>`.
Assigning a name to an existing Group or Dataset
(e.g. ``group['name'] = another_group``) will create a new link in the file
pointing to that object.  Assigning dtypes and NumPy arrays results in
different behavior; see :meth:`Group.__setitem__` for details.

In addition, the following behavior approximates the Python dictionary API:

    - Container syntax (``if name in group``)
    - Iteration yields member names (``for name in group``)
    - Length (``len(group)``)
    - :meth:`listnames <Group.listnames>`
    - :meth:`iternames <Group.iternames>`
    - :meth:`listobjects <Group.listobjects>`
    - :meth:`iterobjects <Group.iterobjects>`
    - :meth:`listitems <Group.listitems>`
    - :meth:`iteritems <Group.iteritems>`

Reference
---------

.. class:: Group

    .. attribute:: name

        Full name of this group in the file (e.g. ``/grp/thisgroup``)

    .. attribute:: attrs

        Dictionary-like object which provides access to this group's
        HDF5 attributes.  See :ref:`attributes` for details.

    .. method:: __getitem__(name) -> Group or Dataset

        Open an object in this group.

    .. method:: __setitem__(name, object)

        Add the given object to the group.

        The action taken depends on the type of object assigned:

        **Named HDF5 object** (Dataset, Group, Datatype)
            A hard link is created in this group which points to the
            given object.

        **Numpy ndarray**
            The array is converted to a dataset object, with default
            settings (contiguous storage, etc.). See :meth:`create_dataset`
            for a more flexible way to do this.

        **Numpy dtype**
            Commit a copy of the datatype as a
            :ref:`named datatype <named_types>` in the file.

        **Anything else**
            Attempt to convert it to an ndarray and store it.  Scalar
            values are stored as scalar datasets. Raise ValueError if we
            can't understand the resulting array dtype.
            
        If a group member of the same name already exists, the assignment
        will fail.

    .. method:: __delitem__(name)

        Remove (unlink) this member.

    .. method:: create_group(name) -> Group

        Create a new HDF5 group.

        Fails with H5Error if the group already exists.

    .. method:: require_group(name) -> Group

        Open the specified HDF5 group, creating it if it doesn't exist.

        Fails with H5Error if an incompatible object (dataset or named type)
        already exists.

    .. method:: create_dataset(name, [shape, [dtype]], [data], **kwds) -> Dataset

        Create a new dataset.  There are two logical ways to specify the dataset:

            1. Give the shape, and optionally the dtype.  If the dtype is not given,
               single-precision floating point ('=f4') will be assumed.
            2. Give a NumPy array (or anything that can be converted to a NumPy array)
               via the "data" argument.  The shape and dtype of this array will be
               used, and the dataset will be initialized to its contents.

        Additional keyword parameters control the details of how the dataset is
        stored.

        **shape** (None or tuple)
            NumPy-style shape tuple.  Required if data is not given.

        **dtype** (None or dtype)
            NumPy dtype (or anything that can be converted).  Optional;
            the default is '=f4'.  Will override the dtype of any data
            array given via the *data* parameter.

        **data** (None or ndarray)
            Either a NumPy ndarray or anything that can be converted to one.

        Keywords:

        **chunks** (None or tuple)
            Manually specify a chunked layout for the dataset.  It's
            recommended you let the library determine this value for you.

        **compression** (None or int)
            Enable DEFLATE (gzip) compression, at this integer value.

        **shuffle** (True/False)
            Enable the shuffle filter, which can provide higher compression ratios
            when used with the compression filter.
        
        **fletcher32** (True/False)
            Enable error detection.

        **maxshape** (None or tuple)
            Make the dataset extendable, up to this maximum shape.  Should be a
            NumPy-style shape tuple.  Dimensions with value None have no upper
            limit.

    .. method:: require_dataset(name, [shape, [dtype]], [data], **kwds) -> Dataset

        Open a new dataset, creating one if it doesn't exist.

        This method operates exactly like :meth:`create_dataset`, except that if
        a dataset with compatible shape and dtype already exists, it is opened
        instead.  The additional keyword arguments are only honored when actually
        creating a dataset; they are ignored for the comparison.

    .. method:: copy(source, dest)

        Recusively copy an object from one location to another, or between files.

        Copies the given object, and (if it is a group) all objects below it in
        the hierarchy.  The destination need not be in the same file.

        **source** (Group, Dataset, Datatype or str)
            Source object or path.

        **dest** (Group or str)
            Destination.  Must be either Group or path.  If a Group object, it may
            be in a different file.

        **Only available with HDF5 1.8.X**

    .. method:: visit(func) -> None or return value from func

        Recursively iterate a callable over objects in this group.

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature::

            func(<member name>) -> <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guranteed.

        Example::

            >>> # List the entire contents of the file
            >>> f = File("foo.hdf5")
            >>> list_of_names = []
            >>> f.visit(list_of_names.append)

        **Only available with HDF5 1.8.X.**

    .. method:: visititems(func) -> None or return value from func

        Recursively visit names and objects in this group and subgroups.

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature::

            func(<member name>, <object>) -> <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guranteed.

        Example::

            # Get a list of all datasets in the file
            >>> mylist = []
            >>> def func(name, obj):
            ...     if isinstance(obj, Dataset):
            ...         mylist.append(name)
            ...
            >>> f = File('foo.hdf5')
            >>> f.visititems(func)

        **Only available with HDF5 1.8.X.**

    .. method:: __len__

        Number of group members

    .. method:: __iter__

        Yields the names of group members

    .. method:: __contains__(name)

        See if the given name is in this group.

    .. method:: listnames

        Get a list of member names

    .. method:: iternames

        Get an iterator over member names.  Equivalent to iter(group).

    .. method:: listobjects

        Get a list with all objects in this group.

    .. method:: iterobjects

        Get an iterator over objects in this group

    .. method:: listitems

        Get an list of (name, object) pairs for the members of this group.

    .. method:: iteritems

        Get an iterator over (name, object) pairs for the members of this group.


Datasets
========

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

Datasets are created using either :meth:`Group.create_dataset` or
:meth:`Group.require_dataset`.  Existing datasets should be retrieved using
the group indexing syntax (``dset = group["name"]``). Calling the constructor
directly is not recommended.

A subset of the NumPy indexing techniques is supported, including the
traditional extended-slice syntax, named-field access, and boolean arrays.
Discrete coordinate selection is also supported via an special indexer class.

Properties
----------

Like Numpy arrays, Dataset objects have attributes named "shape" and "dtype":

    >>> dset.dtype
    dtype('complex64')
    >>> dset.shape
    (4L, 5L)


.. _slicing_access:

Special Features
----------------


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

Sparse selection
----------------

Two mechanisms exist for the case of scattered and/or sparse selection, for
which slab or row-based techniques may not be appropriate.

Boolean "mask" arrays can be used to specify a selection.  The result of
this operation is a 1-D array with elements arranged in the standard NumPy
(C-style) order:

    >>> arr = numpy.arange(100).reshape((10,10))
    >>> dset = f.create_dataset("MyDataset", data=arr)
    >>> result = dset[arr > 50]
    >>> result.shape
    (49,)

If you have a set of discrete points you want to access, you may not want to go
through the overhead of creating a boolean mask.  This is especially the case
for large datasets, where even a byte-valued mask may not fit in memory.  You
can pass a sequence object containing points to the dataset selector via a
custom "CoordsList" instance:

    >>> mycoords = [ (0,0), (3,4), (7,8), (3,5), (4,5) ]
    >>> coords_list = CoordsList(mycoords)
    >>> result = dset[coords_list]
    >>> result.shape
    (5,)

Like boolean-array indexing, the result is a 1-D array.  The order in which
points are selected is preserved.

.. note::
    Boolean-mask and CoordsList indexing rely on an HDF5 construct which
    explicitly enumerates the points to be selected.  It's very flexible but
    most appropriate for 
    reasonably-sized (or sparse) selections.  The coordinate list takes at
    least 8*<rank> bytes per point, and may need to be internally copied.  For
    example, it takes 40MB to express a 1-million point selection on a rank-3
    array.  Be careful, especially with boolean masks.

Special features
----------------

Unlike memory-resident NumPy arrays, HDF5 datasets support a number of optional
features.  These are enabled by the keywords provided to
:meth:`Group.create_dataset`.  Some of the more useful are:

Compression
    Transparent compression 
    (keyword *compression*)
    can substantially reduce the storage space
    needed for the dataset.  The default compression method is GZIP (DEFPLATE),
    which is universally supported by other installations of HDF5.
    Supply an integer between 0 and 9 to enable GZIP compression at that level.
    Using the *shuffle* filter along with this option can improve the
    compression ratio further.

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

    You can also resize a single axis at a time::

        >>> dset.resize(35, axis=1)
        >>> dset.shape
        (20, 35)

    Resizing an array with existing data works differently than in NumPy; if
    any axis shrinks, the data in the missing region is discarded.  Data does
    not "rearrange" itself as it does when resizing a NumPy array.

    .. note::
        Only datasets stored in "chunked" format can be resized.  This format
        is automatically selected when any of the advanced storage options is
        used, or a *maxshape* tuple is provided.  You can also force it to be
        used by specifying ``chunks=True`` at creation time.


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
    >>> f["ScalarDS"][...]
    array(1.0)

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

        GZIP compression level, or None if compression isn't used.

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


.. _attributes:

Attributes
==========

Groups and datasets can have small bits of named information attached to them.
This is the official way to store metadata in HDF5.  Each of these objects
has a small proxy object (:class:`AttributeManager`) attached to it as
``<obj>.attrs``.  This dictionary-like object works like a :class:`Group`
object, with the following differences:

    - Entries may only be scalars and NumPy arrays
    - Each attribute must be small (recommended < 64k for HDF5 1.6)
    - No partial I/O (i.e. slicing) is allowed for arrays

They support the same dictionary API as groups, including the following:

    - Container syntax (``if name in obj.attrs``)
    - Iteration yields member names (``for name in obj.attrs``)
    - Number of attributes (``len(obj.attrs)``)
    - :meth:`listnames <AttributeManager.listnames>`
    - :meth:`iternames <AttributeManager.iternames>`
    - :meth:`listobjects <AttributeManager.listobjects>`
    - :meth:`iterobjects <AttributeManager.iterobjects>`
    - :meth:`listitems <AttributeManager.listitems>`
    - :meth:`iteritems <AttributeManager.iteritems>`

Reference
---------

.. class:: AttributeManager

    .. method:: __getitem__(name) -> NumPy scalar or ndarray

        Retrieve an attribute given a string name.

    .. method:: __setitem__(name, value)

        Set an attribute.  Value must be convertible to a NumPy scalar
        or array.

    .. method:: __delitem__(name)

        Delete an attribute.

    .. method:: __len__

        Number of attributes

    .. method:: __iter__

        Yields the names of attributes

    .. method:: __contains__(name)

        See if the given attribute is present

    .. method:: listnames

        Get a list of attribute names

    .. method:: iternames

        Get an iterator over attribute names

    .. method:: listobjects

        Get a list with all attribute values

    .. method:: iterobjects

        Get an iterator over attribute values

    .. method:: listitems

        Get an list of (name, value) pairs for all attributes.

    .. method:: iteritems

        Get an iterator over (name, value) pairs

.. _named_types:

Named types
===========

There is one last kind of object stored in an HDF5 file.  You can store
datatypes (not associated with any dataset) in a group, simply by assigning
a NumPy dtype to a name::

    >>> group["name"] = numpy.dtype("<f8")

and to get it back::

    >>> named_type = group["name"]
    >>> mytype = named_type.dtype

Objects of this class are immutable and have no methods, just read-only
properties.

Reference
---------

.. class:: Datatype

    .. attribute:: name

        Full name of this object in the HDF5 file (e.g. ``/grp/MyType``)

    .. attribute:: attrs

        Attributes of this object (see :ref:`attributes section <attributes>`)

    .. attribute:: dtype

        NumPy dtype representation of this type

    






