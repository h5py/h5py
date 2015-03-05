Special types
=============

HDF5 supports a few types which have no direct NumPy equivalent.  Among the
most useful and widely used are *variable-length* (VL) types, and enumerated
types.  As of version 2.3, h5py fully supports HDF5 enums and VL types.

How special types are represented
---------------------------------

Since there is no direct NumPy dtype for variable-length strings, enums or
references, h5py extends the dtype system slightly to let HDF5 know how to
store these types.  Each type is represented by a native NumPy dtype, with a
small amount of metadata attached.  NumPy routines ignore the metadata, but
h5py can use it to determine how to store the data.

There are two functions for creating these "hinted" dtypes:

.. function:: special_dtype(**kwds)

    Create a NumPy dtype object containing type hints.  Only one keyword
    may be specified.

    :param vlen: Base type for HDF5 variable-length datatype.

    :param enum: 2-tuple ``(basetype, values_dict)``.  ``basetype`` must be
                 an integer dtype; ``values_dict`` is a dictionary mapping
                 string names to integer values.

    :param ref:  Provide class ``h5py.Reference`` or ``h5py.RegionReference``
                 to create a type representing object or region references
                 respectively.

.. function:: check_dtype(**kwds)

    Determine if the given dtype object is a special type.  Example::

        >>> out = h5py.check_dtype(vlen=mydtype)
        >>> if out is not None:
        ...     print "Vlen of type %s" % out
        str

    :param vlen:    Check for an HDF5 variable-length type; returns base class
    :param enum:    Check for an enumerated type; returns 2-tuple ``(basetype, values_dict)``.
    :param ref:     Check for an HDF5 object or region reference; returns
                    either ``h5py.Reference`` or ``h5py.RegionReference``.


Variable-length strings
-----------------------

In HDF5, data in VL format is stored as arbitrary-length vectors of a base
type.  In particular, strings are stored C-style in null-terminated buffers.
NumPy has no native mechanism to support this.  Unfortunately, this is the
de facto standard for representing strings in the HDF5 C API, and in many
HDF5 applications.

Thankfully, NumPy has a generic pointer type in the form of the "object" ("O")
dtype.  In h5py, variable-length strings are mapped to object arrays.  A
small amount of metadata attached to an "O" dtype tells h5py that its contents
should be converted to VL strings when stored in the file.

Existing VL strings can be read and written to with no additional effort; 
Python strings and fixed-length NumPy strings can be auto-converted to VL
data and stored.

Here's an example showing how to create a VL array of strings::

    >>> f = h5py.File('foo.hdf5')
    >>> dt = h5py.special_dtype(vlen=str)
    >>> ds = f.create_dataset('VLDS', (100,100), dtype=dt)
    >>> ds.dtype.kind
    'O'
    >>> h5py.check_dtype(vlen=ds.dtype)
    <type 'str'>


.. _vlen:

Arbitrary vlen data
-------------------

Starting with h5py 2.3, variable-length types are not restricted to strings.
For example, you can create a "ragged" array of integers::

    >>> dt = h5py.special_dtype(vlen=np.dtype('int32'))
    >>> dset = f.create_dataset('vlen_int', (100,), dtype=dt)
    >>> dset[0] = [1,2,3]
    >>> dset[1] = [1,2,3,4,5]

Single elements are read as NumPy arrays::

    >>> dset[0]
    array([1, 2, 3], dtype=int32)

Multidimensional selections produce an object array whose members are integer
arrays::

    >>> dset[0:2]
    array([array([1, 2, 3], dtype=int32), array([1, 2, 3, 4, 5], dtype=int32)], dtype=object)
    

Enumerated types
----------------

HDF5 has the concept of an *enumerated type*, which is an integer datatype
with a restriction to certain named values.  Since NumPy has no such datatype,
HDF5 ENUM types are read and written as integers.

Here's an example of creating an enumerated type::

    >>> dt = h5py.special_dtype(enum=('i', {"RED": 0, "GREEN": 1, "BLUE": 42}))
    >>> h5py.check_dtype(enum=dt)
    {'BLUE': 42, 'GREEN': 1, 'RED': 0}
    >>> f = h5py.File('foo.hdf5','w')
    >>> ds = f.create_dataset("EnumDS", (100,100), dtype=dt)
    >>> ds.dtype.kind
    'i'
    >>> ds[0,:] = 42
    >>> ds[0,0]
    42
    >>> ds[1,0]
    0

Object and region references
----------------------------

References have their :ref:`own section <refs>`.
