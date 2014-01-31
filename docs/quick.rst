.. _quick:

Quick Start Guide
=================

If you're having trouble installing h5py, refer to :ref:`install`.

Core concepts
-------------

An HDF5 file is a container for two kinds of objects: `datasets`, which are
array-like collections of data, and `groups`, which are folder-like containers
that hold datasets and other groups. The most fundamental thing to remember
when using h5py is:

    **Groups work like dictionaries, and datasets work like NumPy arrays**

The very first thing you'll need to do is create a new file::

    >>> import h5py
    >>> import numpy as np
    >>>
    >>> f = h5py.File("mytestfile.hdf5", "w").

The `File object <hlfile>`_ is your starting point.  It has a couple of
methods which look interesting.  One of them is ``create_dataset``::

    >>> dset = f.create_dataset("mydataset", (100,), dtype='i')

The object we created isn't an array, but `an HDF5 dataset <datasets>`_.
Like NumPy arrays, datasets have both a shape and a data type:

    >>> dset.shape
    (100,)
    >>> dset.dtype
    dtype('int32')

They also support array-style slicing.  This is how you read and write data
from a dataset in the file:

    >>> dset[...] = np.arange(100)
    >>> dset[0]
    0
    >>> dset[10]
    9
    >>> dset[0:100:10]
    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])


Groups and hierarchical organization
------------------------------------

"HDF" stands for "Hierarchical Data Format".  Every object in an HDF5 file
has a name, and they're arranged in a POSIX-style hierarchy with 
``/``-separators::

    >>> dset.name
    u'/mydataset'

The "folders" in this system are called `groups`.  The ``File`` object we
created is itself a group, in this case the `root group`, named ``/``:

    >>> f.name
    u'/'

Creating a subgroup is accomplished via the aptly-named ``create_group``::

    >>> grp = f.create_group("subgroup")

All ``Group`` objects also have the ``create_*`` methods like File::

    >>> dset2 = grp.create_dataset("another_dataset", (50,), dtype='f')
    >>> dset2.name
    u'/subgroup/another_dataset'

By the way, you don't have to create all the intermediate groups manually.
Specifying a full path works just fine::

    >>> dset3 = f.create_dataset('subgroup2/dataset_three', (10,), dtype='i')
    >>> dset3.name
    u'/subgroup2/dataset_three'


Groups are dictionaries (mostly)
--------------------------------

`Groups <groups>`_ support most of the Python dictionary-style interface.  
You retrieve object in the file using the item-retrieval syntax::

    >>> dataset_three = f['subgroup/dataset_three']

Iterating over a group provides the names of its members::

    >>> for name in f:
    ...     print name
    mydataset
    subgroup
    subgroup2

Containership testing also uses names:

    >>> "mydataset" in f
    True
    >>> "somethingelse" in f
    False

You can even use full path names:

    >>> "subgroup/another_dataset" in f
    True

There are also the familiar ``keys()``, ``values()``, ``items()`` and
``iter*()`` methods, as well as ``get()``.

Since iterating over a group only yields its directly-attached members,
iterating over an entire file is accomplished with the ``Group`` methods
``visit()`` and ``visititems()``, which take a callable::

    >>> def printname(name):
    ...     print name
    >>> f.visit(printname)
    mydataset
    subgroup
    subgroup/another_dataset
    subgroup2
    subgroup2/dataset_three


Attributes
----------

One of the best features of HDF5 is that you can store metadata right next
to the data it describes.  All groups and datasets support attached named
bits of data called `attributes`.

Attributes are accessed through the ``attrs`` proxy object, which again
implements the dictionary interface::

    >>> dset.attrs['temperature'] = 99.5
    >>> dset.attrs['temperature']
    99.5
    >>> 'temperature' in dset.attrs
    True


Supported types
---------------

The h5py package supports every Numpy type which maps to a native HDF5 type,
and a few others.

NumPy types:

* Integers: signed/unsigned; 1, 2, 4, 8 bytes; LE/BE
* Floats: 2, 4, 8, 12 bytes; LE/BE
* Structured/compound: may contain arbitrary types, included nested compounds
* Complex numbers: 8, 16, 24 bytes; LE/BE
* Strings: NumPy "S" strings
* Array type: may contain arbitrary types, including nested arrays

Some additional types h5py supports, brought from HDF5:

* Variable-length strings (See also :ref:`strings`)
* Enums
* :ref:`Object and region references <refs>`

For example, variable-length strings let you store Python-style (as opposed to
fixed-width "S") strings using native HDF5 constructs.  No Python-specific
code or pickling is used.

Create a dtype object to represent these by using ``special_dtype``::

    >>> dt = h5py.special_dtype(vlen=str)   # bytes/str/unicode all supported

Then create your dataset using that type:

    >>> dset = f.create_dataset("stringy", (2,), dtype=dt)
    >>> dset[0] = "Hello"
    >>> dset[1] = "Hello this is a longer string"
    >>> dset[...]
    array([Hello, Hello this is a longer string], dtype=object)


