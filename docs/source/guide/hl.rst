
*********
Reference
*********

.. module:: h5py.highlevel

Groups
======

.. class:: Group

    .. method:: __getitem__(name)

        Open an object in this group.

    .. method:: __setitem__(name, object)

        Add the given object to the group.

        The action taken depends on the type of object assigned:

        Named HDF5 object (Dataset, Group, Datatype)
            A hard link is created in this group which points to the
            given object.

        Numpy ndarray
            The array is converted to a dataset object, with default
            settings (contiguous storage, etc.). See :meth:`create_dataset`
            for a more flexible way to do this.

        Numpy dtype
            Commit a copy of the datatype as a named datatype in the file.

        Anything else
            Attempt to convert it to an ndarray and store it.  Scalar
            values are stored as scalar datasets. Raise ValueError if we
            can't understand the resulting array dtype.
            
        If a group member of the same name already exists, the assignment
        will fail.

    .. method:: __delitem__(name)

        Remove (unlink) this member.

    .. method:: __len__

        Number of group members

    .. method:: __iter__

        Yields the names of group members

    .. method:: __contains__(name)

        See if the given name is in this group.

    .. method:: create_group(name) -> Group

        Create a new HDF5 group.

        Fails with H5Error if the group already exists.

    .. method:: require_group(name) -> Group

        Open the specified HDF5 group, creating it if it doesn't exist.

        Fails with H5Error if an incompatible object (dataset or named type)
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

        Keywords:

        **chunks** (None or tuple)
            Manually specify a chunked layout for the dataset.  It's
            recommended you let the library determine this value for you.

        **compression** (None or int)
            Enable DEFLATE (gzip) compression, at this integer value.

        **shuffle** (True/False)
            Enable the shuffle filter, which can provide higher compression ratios
            when used with the compression filter.
        
        **fletcher32** (True/False)
            Enable error detection.

        **maxshape** (None or tuple)
            Make the dataset extendable, up to this maximum shape.  Should be a
            NumPy-style shape tuple.  Dimensions with value None have no upper
            limit.

    .. method:: require_dataset(name, [shape, [dtype]], [data], **kwds) -> Dataset

        Open a new dataset, creating one if it doesn't exist.

        This method operates exactly like :meth:`create_dataset`, except that if
        a dataset with compatible shape and dtype already exists, it is opened
        instead.  The additional keyword arguments are only honored when actually
        creating a dataset; they are ignored for the comparison.

    .. method:: copy(source, dest)

        Recusively copy an object from one location to another, or between files.

        Copies the given object, and (if it is a group) all objects below it in
        the hierarchy.  The destination need not be in the same file.

        **source** (Group, Dataset, Datatype or str)
            Source object or path.

        **dest** (Group or str)
            Destination.  Must be either Group or path.  If a Group object, it may
            be in a different file.

    .. method:: visit(func)

        Recursively iterate a callable over objects in this group.

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature::

            func(<member name>) => <None or return value>

        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guranteed.

        Example::

            >>> # List the entire contents of the file
            >>> f = File("foo.hdf5")
            >>> list_of_names = []
            >>> f.visit(list_of_names.append)

        **Only available with HDF5 1.8.X.**

    .. method:: visititems(func)

        Recursively visit names and objects in this group and subgroups.

        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature::

            func(<member name>, <object>) => <None or return value>

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

        **Only available with HDF5 1.8.X.**

    .. method:: listnames

        Get a list of member names

    .. method:: iternames

        Get an iterator over member names.  Equivalent to iter(group).

    .. method:: listobjects

        Get a list with all objects in this group.

    .. method:: iterobjects

        Get an iterator over objects in this group

    .. method:: listitems

        Get an list of (name, object) pairs for the members of this group.

    .. method:: iteritems

        Get an iterator over (name, object) pairs for the members of this group.











