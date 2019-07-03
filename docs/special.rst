.. _special_types:

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

The metadata h5py attaches to dtypes is not part of the public API,
so it may change between versions.
Use the functions described below to create and check for these types.

Variable-length strings
-----------------------

.. seealso:: :ref:`strings`

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
    >>> dt = h5py.string_dtype(encoding='utf-8')
    >>> ds = f.create_dataset('VLDS', (100,100), dtype=dt)
    >>> ds.dtype.kind
    'O'
    >>> h5py.check_string_dtype(ds.dtype)
    string_info(encoding='utf-8', length=None)

.. function:: string_dtype(encoding='utf-8', length=None)

   Make a numpy dtype for HDF5 strings

   :param encoding: ``'utf-8'`` or ``'ascii'``.
   :param length: ``None`` for variable-length, or an integer for fixed-length
                  string data, giving the length in bytes.

If ``encoding`` is ``'utf-8'``, the variable length strings should be passed as
Python ``str`` objects (``unicode`` in Python 2).
For ``'ascii'``, they should be passed as bytes.

.. function:: check_string_dtype(dt)

   Check if ``dt`` is a string dtype.
   Returns a *string_info* object if it is, or ``None`` if not.

.. class:: string_info

   A named tuple type holding string encoding and length.

   .. attribute:: encoding

      The character encoding associated with the string dtype,
      which can be ``'utf-8'`` or ``'ascii'``.

   .. attribute:: length

      For fixed-length string dtypes, the length in bytes.
      ``None`` for variable-length strings.

.. _vlen:

Arbitrary vlen data
-------------------

Starting with h5py 2.3, variable-length types are not restricted to strings.
For example, you can create a "ragged" array of integers::

    >>> dt = h5py.vlen_dtype(np.dtype('int32'))
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

.. function:: vlen_dtype(basetype)

   Make a numpy dtype for an HDF5 variable-length datatype.

   :param basetype: The dtype of each element in the array.

.. function:: check_vlen_dtype(dt)

   Check if ``dt`` is a variable-length dtype.
   Returns the base type if it is, or ``None`` if not.

Enumerated types
----------------

HDF5 has the concept of an *enumerated type*, which is an integer datatype
with a restriction to certain named values.  Since NumPy has no such datatype,
HDF5 ENUM types are read and written as integers.

Here's an example of creating an enumerated type::

    >>> dt = h5py.enum_dtype({"RED": 0, "GREEN": 1, "BLUE": 42}, basetype='i')
    >>> h5py.check_enum_dtype(dt)
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

.. function:: enum_dtype(values_dict, basetype=np.uint8)

   Create a NumPy representation of an HDF5 enumerated type

   :param values_dict: Mapping of string names to integer values.
   :param basetype: An appropriate integer base dtype large enough to hold the
                    possible options.

.. function:: check_enum_dtype(dt)

   Check if ``dt`` represents an enumerated type.
   Returns the values dict if it is, or ``None`` if not.

Object and region references
----------------------------

References have their :ref:`own section <refs>`.

Older API
---------

Before h5py 2.9, a single pair of functions was used to create and check for
all of these special dtypes. These are still available for backwards
compatibility, but are deprecated in favour of the functions listed above.

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

Custom ``dtype`` s
==================

Usually, :class:`Datatype` objects are created for you on the fly and you do not
have to worry about them. However, in certain cases, namely when no HDF5
equivalent exists for a given :class:`dtype`, you must register the :class:`dtype`
manually for use with h5py.::

    arr = np.array([np.datetime64('2019-06-30')])
    h5py.register_dtype(arr.dtype)
    dset = f.create_dataset("datetimes", data=arr)

.. note::

    It is important to notice that the types registered in this way will only
    be readable be NumPy and compatible tools, and that this format may not be
    universally accepted. In the case of a third party ``dtype``, it will only
    be readable insofar as the ``dtype`` is binary-compatible across old
    versions. In general, opaque datatypes are very sensitive to how you may
    decide to encode your data.

.. function:: register_dtype(dtype dt_in, bytes tag=None)

    Register a NumPy dtype for use with h5py. Types registered in this way
    will be stored as a custom opaque type, with a special tag to map it to
    the corresponding NumPy type.

    Opaque types with this tag will be mapped to NumPy types in the same way.

    The default tag is generated via the code:
    ``b"NUMPY:" + dt_in.descr[0][1].encode()``.

.. function:: deregister_dtype(object obj)

    Deregister a dtype/tag from the NumPy-tag mapping, along with the
    corresponding tag/dtype.
