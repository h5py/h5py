.. _quick:

*****************
Quick Start Guide
*****************

This document is a very quick overview of both HDF5 and h5py.  More
comprehensive documentation is available at:

* :ref:`h5pyreference`

The `HDF Group <http://www.hdfgroup.org>`_ is the final authority on HDF5.
They also have an `introductory tutorial <http://www.hdfgroup.org/HDF5/Tutor/>`_
which provides a good introduction.

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

First, install h5py by following the :ref:`installation instructions <build>`.

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
    <HDF5 file "myfile.hdf5" (mode w, 0 root members)>

In the filesystem metaphor of HDF5, the file object does double duty as the
*root group* (named "/" like its POSIX counterpart).  You can store datasets
in it directly, or create subgroups to keep your data better organized.

Create a dataset
----------------

Datasets are like Numpy arrays which reside on disk; you create them by
providing at least a name and a shape.  Here's an example::

    >>> dset = f.create_dataset("MyDataset", (2,3), '=i4')  # dtype is optional
    >>> dset
    <HDF5 dataset "MyDataset": shape (2, 3), type "<i4">

This creates a new 2-d 6-element (2x3) dataset containing 32-bit signed integer
data, in native byte order, located in the root group at "/MyDataset".

Some familiar NumPy attributes are included::

    >>> dset.shape
    (2, 3)
    >>> dset.dtype
    dtype('int32')

This dataset, like every object in an HDF5 file, has a name::

    >>> dset.name
    '/MyDataset'

If you already have a NumPy array you want to store, just hand it off to h5py::

    >>> arr = numpy.ones((2,3), '=i4')
    >>> dset = f.create_dataset('MyDataset', data=arr)

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

The following slice mechanisms are supported (see :ref:`datasets` for more):

    * Integers/slices (``array[2:11:3]``, etc)
    * Ellipsis indexing (``array[2,...,4:7]``)
    * Simple broadcasting (``array[2]`` is equivalent to ``array[2,...]``)
    * Index lists (``array[ 2, [0,1,4,6] ]``)

along with some emulated advanced indexing features
(see :ref:`sparse_selection`):

    * Boolean array indexing (``array[ array[...] > 0.5 ]``)
    * Discrete coordinate selection (

Closing the file
----------------

You don't need to do anything special to "close" datasets.  However, as with
Python files you should close the file before exiting::

    >>> dset
    <HDF5 dataset "MyDataset": shape (2, 3), type "<i4">
    >>> f.close()
    >>> f
    <Closed HDF5 file>
    >>> dset
    <Closed HDF5 dataset>

H5py tries to close all objects on exit (or when they are no longer referenced),
but it's good practice to close your files anyway.


Groups & multiple objects
=========================

When creating the dataset above, we gave it a name::

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

Groups (including File objects; "f" in this example) support other
dictionary-like operations::

    >>> list(f)                 # iteration
    ['MyDataset', 'SubGroup']
    >>> 'MyDataset' in f        # membership testing
    True
    >>> 'Subgroup/MyOtherDataset' in f      # even for arbitrary paths!
    True
    >>> del f['MyDataset']      # Delete (unlink) a group member

As a safety feature, you can't create an object with a pre-existing name;
you have to manually delete the existing object first::

    >>> grp = f.create_group("NewGroup")
    >>> grp2 = f.create_group("NewGroup")   # wrong
    (H5Error raised)
    >>> del f['NewGroup']
    grp2 = f.create_group("NewGroup")

This restriction reflects HDF5's lack of transactional support, and will not
change.

.. note::

    Most HDF5 versions don't support automatic creation of intermediate
    groups; you can't yet do ``f.create_group('foo/bar/baz')`` unless both
    groups "foo" and "bar" already exist.

Attributes
==========

HDF5 lets you associate small bits of data with both groups and datasets.
This can be used for metadata like descriptive titles or timestamps.

A dictionary-like object which exposes this behavior is attached to every
Group and Dataset object as the attribute ``attrs``.  You can store any scalar
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

Named datatypes
===============

There is in fact one additional, rarely-used kind of object which can be
permanently stored in an HDF5 file.  You can permanently store a *datatype*
object in any group, simply by assigning a NumPy dtype to a name:

    >>> f["MyIntegerDatatype"] = numpy.dtype('<i8')
    >>> htype = f["MyIntegerDatatype"]
    >>> htype
    <HDF5 named type "MyIntegerDatatype" (dtype <i8)>
    >>> htype.dtype
    dtype('int64')

This isn't ordinarily useful because each dataset already carries its own
dtype attribute.  However, if you want to store datatypes which are not used
in any dataset, this is the right way to do it.













