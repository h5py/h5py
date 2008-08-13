*****************
Quick Start Guide
*****************

First, install h5py by following the `installation instructions`__.

__ http://h5py.alfven.org/build.html

The ``import *`` construct is safe when used with the main package::

    >>> from h5py import *

The rest of the examples here assume you've done this.  Among other things, it
imports the three classes ``File``, ``Group`` and ``Dataset``, which will cover
99% of your needs.


Storing simple data
===================

Create a new file
-----------------

Files are opened using a Python-file-like syntax::

    >>> f = File("myfile.hdf5", 'w')    # Create/truncate file
    >>> f
    File "myfile.hdf5", root members:
    >>> type(f)
    <class 'h5py.highlevel.File'>

Create a dataset
----------------

Datasets are like Numpy arrays which reside on disk; they are identified by
a unique name, shape, and a Numpy dtype.  The easiest way to create them is
with a method of the File object you already have::

    >>> dset = f.create_dataset("MyDataset", (2,3), '=i4')
    >>> dset
    Dataset "MyDataset": (2L, 3L) dtype('int32')
    >>> type(dset)
    <class 'h5py.highlevel.Dataset'>

This creates a new 2-d 6-element (2x3) dataset containing 32-bit signed integer
data, in native byte order, located in the file at "/MyDataset".

Read & write data
-----------------

You can now store data in it using the Numpy-like slicing syntax::

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
remember to close the file before exiting Python, to prevent data loss.  This
will automatically close all the open HDF5 objects::

    >>> dset
    Dataset "MyDataset": (2L, 3L) dtype('int32')
    >>> f.close()
    >>> dset
    Invalid dataset


More about datasets
===================

Automatic creation
------------------

If you already have an array you want to store, you don't even need to call
``create_dataset``.  Simply assign it to a name::

    >>> myarr = numpy.ones((50,75))
    >>> f["MyDataset"] = myarr
    >>> f["MyDataset"]
    Dataset "MyDataset": (50L, 75L) dtype('float64')

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
    
You can also access data using Numpy recarray-style indexing.  The following
are all legal slicing syntax for the above array (output omitted for brevity)::

    >>> dset[0,0]
    >>> dset[0,:]
    >>> dset[...]
    >>> dset['temp']
    >>> dset[0,0,'temp']
    >>> dset[8:14:2, ::2, 'voltage']

Shape and data type
-------------------

Like Numpy arrays, Dataset objects have attributes named "shape" and "dtype"::

    >>> dset = f.create_dataset("MyDataset", (4,5), '=c8')
    >>> dset.dtype
    dtype('complex64')
    >>> dset.shape
    (4L, 5L)

These attributes are read-only.

Values and 0-dimensional datasets
---------------------------------

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


Using HDF5 options
------------------

You can specify a number of HDF5 features when creating a dataset.  See the
Dataset constructor for a complete list.  For example, to create a (100,100)
dataset stored as (100,10) size chunks, using GZIP compression level 6::

    >>> dset = f.create_dataset("MyDataset", (100,100), chunks=(100,10), compression=6)


Groups & multiple objects
=========================

The root group
--------------

Like a filesystem, HDF5 supports the concept of storing multiple objects in
containers, called "groups".  The File object behaves as one of these
groups (it's actually the *root group* "``/``", again like a UNIX filesystem).
You store objects by giving them different names:

    >>> f["DS1"] = numpy.ones((2,3))
    >>> f["DS2"] = numpy.ones((1,2))
    >>> f
    File "myfile.hdf5", root members: "DS1", "DS2"

Beware, you need to delete an existing object; as HDF5 won't do this automatically::

    >>> f["DS3"] = numpy.ones((2,2))
    >>> f["DS3"] = numpy.ones((2,2))
    Traceback (most recent call last):
    ... snip traceback ... 
    h5py.h5.DatasetError: Unable to create dataset (H5Dcreate)
    HDF5 Error Stack:
        0: "Unable to create dataset" at H5Dcreate
        1: "Unable to name dataset" at H5D_create
        2: "Already exists" at H5G_insert
        3: "Unable to insert name" at H5G_namei
        4: "Unable to insert entry" at H5G_stab_insert
        5: "Unable to insert key" at H5B_insert
        6: "Can't insert leaf node" at H5B_insert_helper
        7: "Symbol is already present in symbol table" at H5G_node_insert

Removing objects
----------------

You can "delete" (unlink) an object from a group::

    >>> f["DS"] = numpy.ones((10,10))
    >>> f["DS"]
    Dataset "DS": (10L, 10L) dtype('float64')
    >>> del f["DS"]
    >>> f["DS"]
    Traceback (most recent call last):
    ... snip traceback ...
    h5py.h5.ArgsError: Cannot stat object (H5Gget_objinfo)
    HDF5 Error Stack:
        0: "Cannot stat object" at H5Gget_objinfo
        1: "Unable to stat object" at H5G_get_objinfo
        2: "Component not found" at H5G_namei
        3: "Not found" at H5G_stab_find
        4: "Not found" at H5G_node_found

Creating subgroups
------------------

You can create subgroups by giving them names:

    >>> f.create_group('subgrp')
    Group "subgrp" (0 members)
    
Be careful, as most versions of HDF5 don't support "automatic" (recursive)
creation of intermediate groups.  Instead of doing::

    >>> f.create_group('foo/bar/baz')  # WRONG

you have to do:

    >>> f.create_group('foo')
    >>> f.create_group('foo/bar')
    >>> f.create_group('foo/bar/baz')

This restriction will be raised in the future, as HDF5 1.8.X provides a feature
that does this automatically.


Group tricks
------------

Groups support iteration (yields the member names), len() (gives the number
of members), and membership testing:

    >>> g = f.create_group('subgrp')
    >>> g["DS1"] = numpy.ones((2,2))
    >>> g["DS2"] = numpy.ones((1,2))
    >>> g["DS3"] = numpy.ones((10,10))
    >>> for x in g:
    ...     print x
    ...
    DS1
    DS2
    DS3
    >>> for x, ds in g.iteritems():
    ...     print x, ds.shape
    ...
    DS1 (2L, 2L)
    DS2 (1L, 2L)
    DS3 (10L, 10L)
    >>> len(g)
    3
    >>> "DS1" in g
    True
    >>> "DS4" in g
    False

Group caveats
-------------

The HDF5 file graph is not limited to a tree configuration.  Like hard links in
a file system, group "members" are actually references to shared HDF5 objects.
This can lead to odd behavior; for example, it's perfectly legal for a group
to contain itself.  When you assign an existing HDF5 object to a name, HDF5
will create a new reference (hard link) with that name, which points to the
object.

    >>> dset = f.create_dataset("MyDS", (1,2), '=i2')
    >>> f["DS Alias"] = dset   # creates a new hard link

Recursion:

    >>> f["self"] = f
    >>> f.names
    ("self",)
    >>> f["self"].names
    ("self",)
    >>> f["self/self"].names
    ("self",)

While this has many benefits (many paths can share the same underlying data),
you should be careful not to get yourself into trouble.

Attributes
==========

HDF5 lets you associate small bits of data with both groups and datasets.
A dictionary-like object which exposes this behavior is attached to every
Group and Dataset object as the attribute ``attrs``.  You can store any scalar
or array value you like::

    >>> dset = f.create_dataset("MyDS", (2,3), '=i4')
    >>> dset.attrs
    Attributes of "MyDS": (none)
    >>> dset.attrs["Name"] = "My Dataset"
    >>> dset.attrs["Frob Index"] = 4
    >>> dset.attrs["Baz Order"] = numpy.arange(10)
    >>> for name, value in dset.attrs.iteritems():
    ...     print name, value
    ...
    Name My Dataset
    Frob Index 4
    Baz Order [0 1 2 3 4 5 6 7 8 9]

Attributes can be associated with any named HDF5 object, including the root
group. 

More information
================

Everything in h5py is documented with docstrings.  The `online HTML
documentation`__ provides a cross-referenced document with this information.
The classes described in this document are stored in the ``h5py.highlevel``
module.

__ http://h5py.alfven.org/docs/















