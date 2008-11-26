.. _quick:

*****************
Quick Start Guide
*****************

What is HDF5?
=============

It's a filesystem for your data.

Only two kinds of objects are stored in HDF5 files: 
*datasets*, which are homogenous, regular arrays of data (just like
NumPy arrays), and *groups*, which are containers that store datasets and
other groups.  Each file is organized using a filesystem metaphor; groups
are like folders, and datasets are like files.  The syntax for accessing
objects in the file is the traditional POSIX filesystem syntax.  Here
are some examples::

    /                       (Root group)
    /MyGroup                (Subgroup)
    /MyGroup/DS1            (Dataset stored in subgroup)
    /MyGroup/Subgroup/DS2   (and so on)

What is h5py?
=============

It's a simple Python interface to HDF5.  You can interact with files, groups
and datasets using traditional Python and NumPy metaphors.  For example,
groups behave like dictionaries, and datasets have shape and dtype attributes,
and can be sliced and indexed just like real NumPy arrays.  Datatypes are
specified using standard NumPy dtype objects.

You don't need to know anything about the HDF5 library to use h5py, apart from
the basic metaphors of files, groups and datasets.  The library handles all
data conversion transparently, and datasets support advanced features like
efficient multidimensional indexing and nested compound datatypes.

One additional benefit of h5py is that the files it reads and writes are
"plain-vanilla" HDF5 files.  No Python-specific metadata or features are used.
You can read HDF5 files created by any application, and write files that any
HDF5-aware application can understand.

Getting data into HDF5
======================

First, install h5py by following the `installation instructions`__.

__ http://h5py.alfven.org/build.html

The ``import *`` construct is safe when used with the main package::

    >>> from h5py import *

The rest of the examples here assume you've done this.  Among other things, it
imports the three classes ``File``, ``Group`` and ``Dataset``, which will cover
99% of your needs.

Create a new file
-----------------

Files are opened using a Python-file-like syntax::

    >>> f = File("myfile.hdf5", 'w')    # Create/truncate file
    >>> f
    File "myfile.hdf5", root members:
    >>> type(f)
    <class 'h5py.highlevel.File'>

In the filesystem metaphor of HDF5, the file object does double duty as the
*root group* (named "/" like its POSIX counterpart).  You can store datasets
in it directly, or create subgroups to keep your data better organized.

Create a dataset
----------------

Datasets are like Numpy arrays which reside on disk; they are associated with
a name, shape, and a Numpy dtype.  The easiest way to create them is with a
method of the File object you already have::

    >>> dset = f.create_dataset("MyDataset", (2,3), '=i4')
    >>> dset
    Dataset "MyDataset": (2L, 3L) dtype('int32')
    >>> type(dset)
    <class 'h5py.highlevel.Dataset'>

This creates a new 2-d 6-element (2x3) dataset containing 32-bit signed integer
data, in native byte order, located in the root group at "/MyDataset".

Or you can auto-create a dataset from an array, just by giving it a name:

    >>> arr = numpy.ones((2,3), '=i4')
    >>> f["MyDataset"] = arr
    >>> dset = f["MyDataset"]

Shape and dtype information is always available via properties:

    >>> dset.dtype
    dtype('int32')
    >>> dset.shape
    (2L, 3L)

Read & write data
-----------------

You can now store data in it using Numpy-like slicing syntax::

    >>> print dset[...]
    [[0 0 0]
     [0 0 0]]
    >>> import numpy
    >>> myarr = numpy.ones((2,), '=i2')  # The dtype doesn't have to exactly match
    >>> dset[:,0] = myarr
    >>> print dset[...]
    [[1 0 0]
     [1 0 0]]

Closing the file
----------------

You don't need to do anything special to "close" datasets.  However, you must
remember to close the file before exiting Python, to prevent data loss::

    >>> dset
    Dataset "MyDataset": (2L, 3L) dtype('int32')
    >>> f.close()
    >>> dset
    Invalid dataset


Groups & multiple objects
=========================

You've already seen that every object in a file is identified by a name:

    >>> f["DS1"] = numpy.ones((2,3))    # full name "/DS1"
    >>> f["DS2"] = numpy.ones((1,2))    # full name "/DS2"
    >>> f
    File "myfile.hdf5", root members: "DS1", "DS2"

Groups, including the root group ("f", in this example), act somewhat like
Python dictionaries.  They support iteration and membership testing:
    
    >>> list(f)
    ['DS1', 'DS2']
    >>> dict(x, y.shape for x, y in f.iteritems())
    {'DS1': (2,3), 'DS2': (1,2)}
    >>> "DS1" in f
    True
    >>> "FOOBAR" in f
    False

You can "delete" (unlink) an object from a group::

    >>> f["DS"] = numpy.ones((10,10))
    >>> f["DS"]
    Dataset "DS": (10L, 10L) dtype('float64')
    >>> "DS" in f
    True
    >>> del f["DS"]
    >>> "DS" in f
    False

Most importantly, you can create additional subgroups by giving them names:

    >>> g = f.create_group('subgrp')
    >>> g
    Group "subgrp" (0 members)
    >>> g.name
    '/subgrp'
    >>> dset = g.create_dataset("DS3", (10,10))
    >>> dset.name
    '/subgrp/DS3'

Using this feature you can build up an entire virtual filesystem inside an
HDF5 file.  This hierarchical organization is what gives HDF5 its name.

.. note::

    Most HDF5 versions don't support automatic creation of intermediate
    groups; you can't yet do ``f.create_group('foo/bar/baz')`` unless both
    groups "foo" and "bar" already exist.


Attributes
==========

HDF5 lets you associate small bits of data with both groups and datasets.
This can be used for metadata like descriptive titles, timestamps, or any
other purpose you want.

A dictionary-like object which exposes this behavior is attached to every
Group and Dataset object as the attribute ``attrs``.  You can store any scalar
or array value you like::

    >>> dset = f.create_dataset("MyDS", (2,3), '=i4')
    >>> dset.attrs
    Attributes of "MyDS": (none)
    >>> dset.attrs["Name"] = "My Dataset"
    >>> dset.attrs["Frob Index"] = 4
    >>> dset.attrs["Order Array"] = numpy.arange(10)
    >>> for name, value in dset.attrs.iteritems():
    ...     print name+":", value
    ...
    Name: My Dataset
    Frob Index: 4
    Order Array: [0 1 2 3 4 5 6 7 8 9]


Named datatypes
===============

There is in fact one additional, rarely-used kind of object which can be
permanently stored in an HDF5 file.  You can permanently store a *datatype*
object in any group, simply by assigning a NumPy dtype to a name:

    >>> f["MyIntegerDatatype"] = numpy.dtype('<i8')
    >>> htype = f["MyIntegerDatatype"]
    >>> htype.dtype
    dtype('int64')

This isn't ordinarily useful because each dataset already carries its own
dtype attribute.  However, if you want to store datatypes which are not used
in any dataset, this is the right way to do it.

More information
================

See the :ref:`reference chapter <h5pyreference>` for complete documentation of
high-level interface objects like Groups and Datasets.

The `HDF Group`__ is the final authority on HDF5.  Their `user
manual`__ is a great introduction to the basic concepts of HDF5, albeit from
the perspective of a C programmer.

__ http://www.hdfgroup.org/HDF5/
__ http://www.hdfgroup.org/HDF5/doc/UG/index.html














