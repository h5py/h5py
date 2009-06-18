=====================
Special types in h5py
=====================

HDF5 supports a few types which have no direct NumPy equivalent.  Among the
most useful and widely used are *variable-length* (VL) types, and enumerated
types.  As of version 1.2, h5py fully supports HDF5 enums, and has partial
support for VL types.

Variable-length strings
-----------------------

In HDF5, data in VL format is stored as arbitrary-length vectors of a base
type.  In particular, strings are stored C-style in null-terminated buffers.
NumPy has no native mechanism to support this.  Unfortunately, this is the
de facto standard for representing strings in the HDF5 C API, and in many
HDF5 applications.

Thankfully, NumPy has a generic pointer type in the form of the "object" ("O")
dtype.  In h5py 1.2, variable-length strings are mapped to object arrays.  A
small amount of metadata attached to an "O" dtype tells h5py that its contents
should be converted to VL strings when stored in the file.


Existing VL strings can be read and written to with no additional effort; 
Python strings and fixed-length NumPy strings can be auto-converted to VL
data and stored.  However, creating VL data requires the use of a special
"hinted" dtype object.  Two functions are provided at the package level for
this purpose:

.. function:: h5py.new_vlen(basetype) -> dtype

    Create a new object dtype which represents a VL type.  Currently
    *basetype* must be the Python string type (str).

.. function:: h5py.get_vlen(dtype) -> dtype or None

    Get the base type of a variable-length dtype, or None if *dtype*
    doesn't represent a VL type.

Here's an example showing how to create a VL array of strings::

    >>> f = h5py.File('foo.hdf5')
    >>> dt = h5py.new_vlen(str)
    >>> ds = f.create_dataset('VLDS', (100,100), dtype=dt)
    >>> ds.dtype.kind
    ... 'O'
    >>> h5py.get_vlen(ds.dtype)
    ... <type 'str'>

Enumerated types
----------------

HDF5 has the concept of an *enumerated type*, which is an integer datatype
with a restriction to certain named values.  Since NumPy has no such datatype,
HDF5 ENUM types are read and written as integers.  Like variable-length
strings, you can create a new enumerated type from a NumPy integer base type
by using convenience functions to attach a small amount of metadata:

.. function:: h5py.new_enum(dtype, values) -> dtype

    Create a new enumerated type, from a NumPy integer dtype and a dictionary
    of {"name": value} pairs.  Keys must be strings, and values must be
    integers.

.. function:: h5py.get_enum(dtype) -> dict or None

    Extract the name/value dictionary from an existing enum dtype.  Returns
    None if the dtype does not contain metadata.

Here's an example::

    >>> dt = h5py.new_enum('i', {"RED": 0, "GREEN": 1, "BLUE": 42})
    >>> h5py.get_enum(dt)
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









