.. _quick:

Quick Start Guide
=================

This document is a very quick overview of both HDF5 and h5py.  You may also
want to consult `the h5py FAQ (at Google Code) <http://code.google.com/p/h5py/wiki/FAQ>`_.

The `HDF Group <http://www.hdfgroup.org>`_ is the final authority on HDF5.
They also have an `introductory tutorial <http://www.hdfgroup.org/HDF5/Tutor/>`_
which provides a good overview.

What is HDF5?
-------------

It's a filesystem for your data.

More accurately, it's a widely used scientific file format for archiving and
sharing large amounts of numerical data.  HDF5 files contain *datasets*, which
are homogenous, regular arrays of data, like NumPy arrays, and *groups*,
which are containers that store datasets and other groups.

In this sense, the structure of an HDF5 file is analagous to a POSIX filesystem.
In fact, this is exactly the syntax used by HDF5 itself to locate resources::

    /                       (Root group)
    /MyGroup                (Subgroup)
    /MyGroup/DS1            (Dataset stored in subgroup)
    /MyGroup/Subgroup/DS2   (and so on)

HDF5 also has a well-developed type system, supporting integers and floats
of all the normal sizes and byte orders, as well as more advanced constructs
like compound and array types.  The library handles all type conversion
internally; you can read and write data without having to worry about things
like endian-ness or precision.

What is h5py?
-------------

It's a simple Python interface to HDF5.  You can interact with files, groups
and datasets using traditional Python and NumPy metaphors.  For example,
groups behave like dictionaries, and datasets have shape and dtype attributes,
and can be sliced and indexed just like real NumPy arrays.  Datatypes are
specified using standard NumPy dtype objects.

You don't need to know anything about the HDF5 library to use h5py, apart from
the basic metaphors of files, groups and datasets.  The library handles all
data conversion transparently, and translates operations like slicing into
the appropriate efficient HDF5 routines.

One additional benefit of h5py is that the files it reads and writes are
"plain-vanilla" HDF5 files.  No Python-specific metadata or features are used.
You can read files created by most HDF5 applications, and write files that
any HDF5-aware application can understand.

Getting data into HDF5
----------------------

First, install h5py by following the :ref:`installation instructions <build>`.

Since examples are better than long-winded explanations, here's how to:

    * Make a new file
    * Create an integer dataset, with shape (100,100)
    * Initialize the dataset to the value 42
    * Close the file

    >>> import h5py
    >>> f = h5py.File('myfile.hdf5')
    >>> dset = f.create_dataset("MyDataset", (100, 100), 'i')
    >>> dset[...] = 42
    >>> f.close()

The :ref:`File <hlfile>` constructor accepts modes similar to Python file modes,
including "r", "w", and "a" (the default):

    >>> f = h5py.File('file1.hdf5', 'w')    # overwrite any existing file
    >>> f = h5py.File('file2.hdf5', 'r')    # open read-only

The :ref:`Dataset <datasets>` object ``dset`` above represents a new 2-d HDF5
dataset.  Some features will be familiar to NumPy users::

    >>> dset.shape
    (100, 100)
    >>> dset.dtype
    dtype('int32')

You can even automatically create a dataset from an existing array:

    >>> import numpy as np
    >>> arr = np.ones((2,3), '=i4')
    >>> dset = f.create_dataset('AnotherDataset', data=arr)

HDF5 datasets support many other features, like chunking and transparent 
compression.  See the section "ref:`datasets` for more info.

Getting your data back
^^^^^^^^^^^^^^^^^^^^^^

You can store and retrieve data using Numpy-like slicing syntax.  The following
slice mechanisms are supported:

    * Integers/slices (``array[2:11:3]``, etc)
    * Ellipsis indexing (``array[2,...,4:7]``)
    * Simple broadcasting (``array[2]`` is equivalent to ``array[2,...]``)
    * Index lists (``array[ 2, [0,1,4,6] ]``)

along with some emulated advanced indexing features
(see :ref:`sparse_selection`):

    * Boolean array indexing (``array[ array[...] > 0.5 ]``)

Closing the file
^^^^^^^^^^^^^^^^

You don't need to do anything special to "close" datasets or groups when you're
done with them  However, as with Python files you should close the file
before exiting::

    >>> f.close()

Groups & multiple objects
-------------------------

When creating the dataset in the first example, we gave it the name
"MyDataset".  The Python property ".name" lets us look this up:

    >>> dset.name
    '/MyDataset'

This bears a suspicious resemblance to a POSIX filesystem path; in this case,
we say that MyDataset resides in the *root group* (``/``) of the file.  You
can create other groups as well::

    >>> subgroup = f.create_group("SubGroup")
    >>> subgroup.name
    '/SubGroup'

They can in turn contain new datasets or additional groups::

    >>> dset2 = subgroup.create_dataset('MyOtherDataset', (4,5), '=f8')
    >>> dset2.name
    '/SubGroup/MyOtherDataset'

You can access the contents of groups using dictionary-style syntax, using
POSIX-style paths::

    >>> dset2 = subgroup['MyOtherDataset']
    >>> dset2 = f['/SubGroup/MyOtherDataset']   # equivalent

The canny reader will have noticed that File objects support the same
operations as Group objects.  In fact, File is implemented as a subclass
of Group.  This reflects the long-term principle in the HDF5 C API that files
and groups are largely interchangable.

Groups support other dictionary-like operations::

    >>> list(f)
    ['MyDataset', 'SubGroup']
    >>> 'MyDataset' in f
    True
    >>> 'Subgroup/MyOtherDataset' in f
    True
    >>> del f['MyDataset']

Unlike dictionaries, you can't create an object with a pre-existing name;
you have to manually delete the existing object first::

    >>> grp = f.create_group("NewGroup")
    >>> grp = f.create_group("NewGroup")
    ValueError: Name already exists (Symbol table: Object already exists)
    >>> del f['NewGroup']
    >>> grp = f.create_group("NewGroup")

Attributes
----------

HDF5 lets you associate small bits of data with both groups and datasets.
This can be used for metadata like descriptive titles or timestamps.

A dictionary-like object which exposes this behavior is attached to every
Group and Dataset object as property ``attrs``.  You can store any scalar
or array value you like::

    >>> dset.attrs
    <Attributes of HDF5 object "MyDataset" (0)>
    >>> dset.attrs["Name"] = "My Dataset"
    >>> dset.attrs["Frob Index"] = 4
    >>> dset.attrs["Order Array"] = numpy.arange(10)
    >>> for name, value in dset.attrs.iteritems():
    ...     print name+":", value
    ...
    Name: My Dataset
    Frob Index: 4
    Order Array: [0 1 2 3 4 5 6 7 8 9]

Attribute proxy objects support the same dictionary-like API as groups, but
unlike group members, you can directly overwrite existing attributes:

    >>> dset.attrs["Name"] = "New Name"

Other stuff
-----------

In addition to this basic behavior, HDF5 has a lot of other goodies.  Some
of these features are:

* :ref:`Compressed datasets <dsetfeatures>`
* :ref:`Soft and external links <softlinks>`
* :ref:`Object and region references <refs>`













