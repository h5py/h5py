Special types
=============

HDF5 supports a few types which have no direct NumPy equivalent.  Among the
most useful and widely used are *variable-length* (VL) types, and enumerated
types.  As of version 1.2, h5py fully supports HDF5 enums, and has partial
support for VL types.

How special types are represented
---------------------------------

Since there is no direct NumPy dtype for variable-length strings, enums or
references, h5py extends the dtype system slightly to let HDF5 know how to
store these types.  Each type is represented by a native NumPy dtype, with a
small amount of metadata attached.  NumPy routines ignore the metadata, but
h5py can use it to determine how to store the data.

There are two functions for creating these "hinted" dtypes:

.. autofunction:: h5py.special_dtype

.. autofunction:: h5py.check_dtype

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

References
----------

References have their :ref:`own section <refs>`.
