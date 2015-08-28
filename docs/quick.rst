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

Suppose someone has sent you a HDF5 file, :code:`mytestfile.hdf5`. (To create this file, read `Appendix: Creating a file`_.) The very first thing you'll need to do is to open the file for reading::

    >>> import h5py
    >>> f = h5py.File('mytestfile.hdf5', 'r')

The :ref:`File object <file>` is your starting point. What is stored in this file? Remember :py:class:`h5py.File` acts like a Python dictionary, thus we can check the keys,

    >>> f.keys()
    [u'mydataset']

Based on our observation, there is one data set, :code:`mydataset` in the file. 
Let us examine the data set as a :ref:`Dataset <dataset>` object

    >>> dset = f['mydataset']

The object we obtained isn't an array, but :ref:`an HDF5 dataset <dataset>`.
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

For more, see :ref:`file` and :ref:`dataset`.

Appendix: Creating a file
+++++++++++++++++++++++++

At this point, you may wonder how :code:`mytestdata.hdf5` is created. 
We can create a file by setting the :code:`mode` to :code:`w` when 
the File object is initialized. Some other modes are :code:`a` 
(for read/write/create access), and 
:code:`r+` (for read/write access). 
A full list of file access modes and their meanings is at :ref:`file`. ::  

    >>> import h5py
    >>> import numpy as np
    >>> f = h5py.File("mytestfile.hdf5", "w")

The :ref:`File object <file>` has a couple of methods which look interesting. One of them is ``create_dataset``, which 
as the name suggests, creates a data set of given shape and dtype ::

    >>> dset = f.create_dataset("mydataset", (100,), dtype='i')

Now we can flush the file to the disk

    >>> f.flush()
    >>> del f

The File object is a context manager; thus we can create the file and
manage the :code:`flush` operation with the following lines ::

    >>> import h5py
    >>> import numpy as np
    >>> with h5py.File("mytestfile.hdf5", "w") as f:
    >>>     dset = f.create_dataset("mydataset", (100,), dtype='i')
   
                
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

Creating a subgroup is accomplished via the aptly-named ``create_group``. But we need to open the file in read/write mode first ::
   
    >>> f = h5py.File('mydataset.hdf5', 'r+')
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

Containership testing also uses names:

    >>> "mydataset" in f
    True
    >>> "somethingelse" in f
    False

You can even use full path names:

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


