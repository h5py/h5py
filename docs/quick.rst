.. _quick:

Quick Start Guide
=================

Install
-------

With `Anaconda <http://continuum.io/downloads>`_ or 
`Miniconda <http://conda.pydata.org/miniconda.html>`_::

    conda install h5py
    
With `Enthought Canopy <https://www.enthought.com/products/canopy/>`_, use
the GUI package manager or::

    enpkg h5py

With pip or setup.py, see :ref:`install`.

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
    >>> f = h5py.File("mytestfile.hdf5", "w")

The :ref:`File object <file>` is your starting point.  It has a couple of
methods which look interesting.  One of them is ``create_dataset``::

    >>> dset = f.create_dataset("mydataset", (100,), dtype='i')

The object we created isn't an array, but :ref:`an HDF5 dataset <dataset>`.
Like NumPy arrays, datasets have both a shape and a data type::

    >>> dset.shape
    (100,)
    >>> dset.dtype
    dtype('int32')

They also support array-style slicing.  This is how you read and write data
from a dataset in the file::

    >>> dset[...] = np.arange(100)
    >>> dset[0]
    0
    >>> dset[10]
    10
    >>> dset[0:100:10]
    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

For more, see :ref:`file` and :ref:`dataset`.


Groups and hierarchical organization
------------------------------------

"HDF" stands for "Hierarchical Data Format".  Every object in an HDF5 file
has a name, and they're arranged in a POSIX-style hierarchy with 
``/``-separators::

    >>> dset.name
    u'/mydataset'

The "folders" in this system are called :ref:`groups <group>`.  The ``File`` object we
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

Groups support most of the Python dictionary-style interface.  
You retrieve objects in the file using the item-retrieval syntax::

    >>> dataset_three = f['subgroup2/dataset_three']

Iterating over a group provides the names of its members::

    >>> for name in f:
    ...     print name
    mydataset
    subgroup
    subgroup2

Membership testing also uses names::

    >>> "mydataset" in f
    True
    >>> "somethingelse" in f
    False

You can even use full path names::

    >>> "subgroup/another_dataset" in f
    True

There are also the familiar ``keys()``, ``values()``, ``items()`` and
``iter()`` methods, as well as ``get()``.

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

For more, see :ref:`group`.

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

For more, see :ref:`attributes`.


