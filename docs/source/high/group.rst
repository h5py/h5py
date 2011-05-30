=============
Group Objects
=============

Creating and using groups
-------------------------

Groups are the container mechanism by which HDF5 files are organized.  From
a Python perspective, they operate somewhat like dictionaries.  In this case
the "keys" are the names of group members, and the "values" are the members
themselves (:class:`Group` and :class:`Dataset`) objects.

Group objects also contain most of the machinery which makes HDF5 useful.
The :ref:`File object <hlfile>` does double duty as the HDF5 *root group*, and
serves as your entry point into the file:

    >>> f = h5py.File('foo.hdf5','w')
    >>> f.name
    '/'
    >>> f.keys()
    []

New groups are easy to create:

    >>> grp = f.create_group("bar")
    >>> grp.name
    '/bar'
    >>> subgrp = grp.create_group("baz")
    >>> subgrp.name
    '/bar/baz'

Datasets are also created by a Group method:

    >>> dset = subgrp.create_dataset("MyDS", (100,100), dtype='i')
    >>> dset.name
    '/bar/baz/MyDS'

Accessing objects
-----------------

Groups implement a subset of the Python dictionary convention.  They have
methods like ``keys()``, ``values()`` and support iteration.  Most importantly,
they support the indexing syntax, and standard exceptions:

    >>> myds = subgrp["MyDS"]
    >>> missing = subgrp["missing"]
    KeyError: "Name doesn't exist (Symbol table: Object not found)"

Objects can be deleted from the file using the standard syntax::

    >>> del subgroup["MyDataset"]

Group objects implement the following subset of the Python "mapping" interface:

- Container syntax: ``if name in group``
- Iteration; yields member names: ``for name in group``
- Length: ``len(group)``
- :meth:`keys() <Group.keys>` 
- :meth:`values() <Group.values>`
- :meth:`items() <Group.items>`
- :meth:`iterkeys() <Group.iterkeys>`
- :meth:`itervalues() <Group.itervalues>`
- :meth:`iteritems() <Group.iteritems>`
- :meth:`__setitem__() <Group.__setitem__>`
- :meth:`__getitem__() <Group.__getitem__>`
- :meth:`__delitem__() <Group.__delitem__>`
- :meth:`get() <Group.get>`

Python 3 dict interface
-----------------------

When using h5py from Python 3, the keys(), values() and items() methods
will return view-like objects instead of lists.  These objects support
containership testing and iteration, but can't be sliced like lists.

The iterkeys(), itervalues(), and iteritems() methods are likewise not
available in Python 3.  You may wish to use the standard conversion script
2to3 which ships with Python to accomodate these changes.

.. _softlinks:

Soft links
----------

Like a UNIX filesystem, HDF5 groups can contain "soft" or symbolic links,
which contain a text path instead of a pointer to the object itself.  You
can easily create these in h5py:

    >>> myfile = h5py.File('foo.hdf5','w')
    >>> group = myfile.create_group("somegroup")
    >>> myfile["alias"] = h5py.SoftLink('/somegroup')

Once created, soft links act just like regular links.  You don't have to
do anything special to access them:

    >>> print myfile["alias"]
    <HDF5 group "/alias" (0 members)>

However, they "point to" the target:

    >>> myfile['alias'] == myfile['somegroup']
    True

If the target is removed, they will "dangle":

    >>> del myfile['somegroup']
    >>> print myfile['alias']
    KeyError: 'Component not found (Symbol table: Object not found)'

.. note::

    The class h5py.SoftLink doesn't actually do anything by itself; it only
    serves as an indication to the Group object that you want to create a
    soft link.


External links
--------------

New in HDF5 1.8, external links are "soft links plus", which allow you to
specify the name of the file as well as the path to the desired object.  You
can refer to objects in any file you wish.  Use similar syntax as for soft
links:

    >>> myfile = h5py.File('foo.hdf5','w')
    >>> myfile['ext link'] = h5py.ExternalLink("otherfile.hdf5", "/path/to/resource")

When the link is accessed, the file "otherfile.hdf5" is opened, and object at
"/path/to/resource" is returned.

.. note::

    Since the object retrieved is in a different file, its ".file" and ".parent"
    properties will refer to objects in that file, *not* the file in which the
    link resides.

Getting info on links
---------------------

Although soft and external links are designed to be transparent, there are some
cases where it is valuable to know when they are in use.  The Group method
"get" takes keyword arguments which let you choose whether to follow a link or
not, and to return the class of link in use (soft or external).

Reference
---------

.. autoclass:: h5py.Group
   
    **``Group`` methods**

    .. automethod:: h5py.Group.__setitem__
    .. automethod:: h5py.Group.__getitem__

    .. automethod:: h5py.Group.create_group
    .. automethod:: h5py.Group.create_dataset

    .. automethod:: h5py.Group.require_group
    .. automethod:: h5py.Group.require_dataset

    .. automethod:: h5py.Group.copy
    .. automethod:: h5py.Group.visit
    .. automethod:: h5py.Group.visititems

    **Dictionary-like methods**

    .. automethod:: h5py.Group.keys
    .. automethod:: h5py.Group.values
    .. automethod:: h5py.Group.items

    .. automethod:: h5py.Group.iterkeys
    .. automethod:: h5py.Group.itervalues
    .. automethod:: h5py.Group.iteritems

    .. automethod:: h5py.Group.get

    **Properties common to all HDF5 objects:**

    .. autoattribute:: h5py.Group.file
    .. autoattribute:: h5py.Group.parent
    .. autoattribute:: h5py.Group.name
    .. autoattribute:: h5py.Group.id
    .. autoattribute:: h5py.Group.ref
    .. autoattribute:: h5py.Group.attrs





