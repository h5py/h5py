=============
Group Objects
=============

Creating and using groups
-------------------------

Groups are the container mechanism by which HDF5 files are organized.  From
a Python perspective, they operate somewhat like dictionaries.  In this case
the "keys" are the names of group entries, and the "values" are the entries
themselves (:class:`Group` and :class:`Dataset`) objects.

Group objects also contain most of the machinery which makes HDF5 useful.
The :ref:`File object <hlfile>` does double duty as the HDF5 `root group`, and
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

Reference
---------

.. class:: Group

    .. attribute:: name

        Full name of this group in the file (e.g. ``/grp/thisgroup``)

    .. attribute:: attrs

        Dictionary-like object which provides access to this group's
        HDF5 attributes.  See :ref:`attributes` for details.

    .. attribute:: file
        
        The ``File`` instance used to open this HDF5 file.

    .. attribute:: parent

        A group which contains this object, according to dirname(obj.name).

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
            Commit a copy of the datatype as a named type in the file.

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

        Fails with ValueError if the group already exists.

    .. method:: require_group(name) -> Group

        Open the specified HDF5 group, creating it if it doesn't exist.

        Fails with TypeError if an incompatible object (dataset or named type)
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

        Keywords (see :ref:`dsetfeatures`):

        **chunks** (None, True or shape tuple)
            Store the dataset in chunked format.  Automatically
            selected if any of the other keyword options are given.  If you
            don't provide a shape tuple, the library will guess one for you.
            Chunk sizes of 100kB-300kB work best with HDF5. 

        **compression** (None, string ["gzip" | "lzf" | "szip"] or int 0-9)
            Enable dataset compression.  DEFLATE, LZF and (where available)
            SZIP are supported.  An integer is interpreted as a GZIP level
            for backwards compatibility

        **compression_opts** (None, or special value)
            Setting for compression filter; legal values for each filter
            type are:

            ======      ======================================
            "gzip"      Integer 0-9
            "lzf"       (none allowed)
            "szip"      2-tuple ('ec'|'nn', even integer 0-32)
            ======      ======================================

            See the ``filters`` module for a detailed description of each
            of these filters.

        **shuffle** (True/False)
            Enable/disable data shuffling, which can improve compression
            performance.

        **fletcher32** (True/False)
            Enable Fletcher32 error detection; may be used with or without
            compression.

        **maxshape** (None or shape tuple)
            Make the dataset extendable, up to this maximum shape.  Should be a
            NumPy-style shape tuple.  Dimensions with value None have no upper
            limit.

    .. method:: require_dataset(name, [shape, [dtype]], [data], **kwds) -> Dataset

        Open a new dataset, creating one if it doesn't exist.

        This method operates exactly like :meth:`create_dataset`, except that if
        a dataset with compatible shape and dtype already exists, it is opened
        instead.  The additional keyword arguments are only honored when actually
        creating a dataset; they are ignored for the comparison.

        If an existing incompatible object (Group or Datatype) already exists
        with the given name, fails with ValueError.

    .. method:: copy(source, dest, name=None)

        **Only available with HDF5 1.8**

        Recusively copy an object from one location to another, or between files.

        Copies the given object, and (if it is a group) all objects below it in
        the hierarchy.  The destination need not be in the same file.

        **source** (Group, Dataset, Datatype or str)
            Source object or path.

        **dest** (Group or str)
            Destination.  Must be either Group or path.  If a Group object, it may
            be in a different file.

        **name** (None or str)
            If the destination is a Group object, you can override the name
            for the newly created member.  Otherwise a new name will be chosen
            using basename(source.name).

    .. method:: visit(func) -> None or return value from func

        **Only available with HDF5 1.8**

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

    .. method:: visititems(func) -> None or return value from func

        **Only available with HDF5 1.8**

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

    .. method:: __len__

        Number of group members

    .. method:: __iter__

        Yields the names of group members

    .. method:: __contains__(name)

        See if the given name is in this group.

    .. method:: keys

        Get a list of member names

    .. method:: iterkeys

        Get an iterator over member names.  Equivalent to iter(group).

    .. method:: values

        Get a list with all objects in this group.

    .. method:: itervalues

        Get an iterator over objects in this group

    .. method:: items

        Get an list of (name, object) pairs for the members of this group.

    .. method:: iteritems

        Get an iterator over (name, object) pairs for the members of this group.

    .. method:: get(name, default)

        Retrieve the member, or *default* if it doesn't exist.

