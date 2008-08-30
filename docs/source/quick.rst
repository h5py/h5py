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


Getting data into HDF5
======================

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

(Main chapter: :ref:`Datasets`)

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


Groups & multiple objects
=========================

Like a filesystem, HDF5 supports the concept of storing multiple objects in
containers, called "groups".  The File object behaves as one of these
groups (it's actually the *root group* "``/``", again like a UNIX filesystem).
You store objects by giving them different names:

    >>> f["DS1"] = numpy.ones((2,3))
    >>> f["DS2"] = numpy.ones((1,2))
    >>> f
    File "myfile.hdf5", root members: "DS1", "DS2"

As with other Python container objects, they support iteration and membership
testing:
    
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

You create additional subgroups by giving them names:

    >>> f.create_group('subgrp')
    Group "subgrp" (0 members)
    
.. note::

    Most HDF5 versions don't support automatic creation of intermediate
    groups; you can't yet do ``f.create_group('foo/bar/baz')``.


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















