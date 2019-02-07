Module H5T
==========

.. automodule:: h5py.h5t

Functions specific to h5py
--------------------------

.. autofunction:: py_create
.. autofunction:: string_dtype
.. autofunction:: check_string_dtype
.. autofunction:: vlen_dtype
.. autofunction:: check_vlen_dtype
.. autofunction:: enum_dtype
.. autofunction:: check_enum_dtype
.. autofunction:: special_dtype
.. autofunction:: check_dtype

Functional API
--------------
.. autofunction:: create
.. autofunction:: open
.. autofunction:: array_create
.. autofunction:: enum_create
.. autofunction:: vlen_create
.. autofunction:: decode
.. autofunction:: convert
.. autofunction:: find

Type classes
------------

.. autoclass:: TypeID
    :members:

Atomic classes
~~~~~~~~~~~~~~

Atomic types are integers and floats.  Much of the functionality for each is
inherited from the base class :class:`TypeAtomicID`.

.. autoclass:: TypeAtomicID
    :show-inheritance:
    :members:

.. autoclass:: TypeIntegerID
    :show-inheritance:
    :members:

.. autoclass:: TypeFloatID
    :show-inheritance:
    :members:

Strings
~~~~~~~

.. autoclass:: TypeStringID
    :show-inheritance:
    :members:

Compound Types
~~~~~~~~~~~~~~

Traditional compound type (like NumPy record type) and enumerated types share
a base class, :class:`TypeCompositeID`.

.. autoclass:: TypeCompositeID
    :show-inheritance:
    :members:

.. autoclass:: TypeCompoundID
    :show-inheritance:
    :members:

.. autoclass:: TypeEnumID
    :show-inheritance:
    :members:

Other types
~~~~~~~~~~~

.. autoclass:: TypeArrayID
    :show-inheritance:
    :members:

.. autoclass:: TypeOpaqueID
    :show-inheritance:
    :members:

.. autoclass:: TypeVlenID
    :show-inheritance:
    :members:

.. autoclass:: TypeBitfieldID
    :show-inheritance:
    :members:

.. autoclass:: TypeReferenceID
    :show-inheritance:
    :members:

Predefined Datatypes
--------------------

These locked types are pre-allocated by the library.

Floating-point
~~~~~~~~~~~~~~

.. data:: IEEE_F32LE
.. data:: IEEE_F32BE
.. data:: IEEE_F64LE
.. data:: IEEE_F64BE

Integer types
~~~~~~~~~~~~~

.. data:: STD_I8LE
.. data:: STD_I16LE
.. data:: STD_I32LE
.. data:: STD_I64LE

.. data:: STD_I8BE
.. data:: STD_I16BE
.. data:: STD_I32BE
.. data:: STD_I64BE

.. data:: STD_U8LE
.. data:: STD_U16LE
.. data:: STD_U32LE
.. data:: STD_U64LE

.. data:: STD_U8BE
.. data:: STD_U16BE
.. data:: STD_U32BE
.. data:: STD_U64BE

.. data:: NATIVE_INT8
.. data:: NATIVE_UINT8
.. data:: NATIVE_INT16
.. data:: NATIVE_UINT16
.. data:: NATIVE_INT32
.. data:: NATIVE_UINT32
.. data:: NATIVE_INT64
.. data:: NATIVE_UINT64
.. data:: NATIVE_FLOAT
.. data:: NATIVE_DOUBLE

Reference types
~~~~~~~~~~~~~~~

.. data:: STD_REF_OBJ
.. data:: STD_REF_DSETREG

String types
~~~~~~~~~~~~

.. data:: C_S1

    Null-terminated fixed-length string

.. data:: FORTRAN_S1

    Zero-padded fixed-length string

.. data:: VARIABLE

    Variable-length string

Python object type
~~~~~~~~~~~~~~~~~~

.. data:: PYTHON_OBJECT

Module constants
----------------

Datatype class codes
~~~~~~~~~~~~~~~~~~~~

.. data:: NO_CLASS
.. data:: INTEGER
.. data:: FLOAT
.. data:: TIME
.. data:: STRING
.. data:: BITFIELD
.. data:: OPAQUE
.. data:: COMPOUND
.. data:: REFERENCE
.. data:: ENUM
.. data:: VLEN
.. data:: ARRAY

API Constants
~~~~~~~~~~~~~

.. data:: SGN_NONE
.. data:: SGN_2

.. data:: ORDER_LE
.. data:: ORDER_BE
.. data:: ORDER_VAX
.. data:: ORDER_NONE
.. data:: ORDER_NATIVE

.. data:: DIR_DEFAULT
.. data:: DIR_ASCEND
.. data:: DIR_DESCEND

.. data:: STR_NULLTERM
.. data:: STR_NULLPAD
.. data:: STR_SPACEPAD

.. data:: NORM_IMPLIED
.. data:: NORM_MSBSET
.. data:: NORM_NONE

.. data:: CSET_ASCII
.. DATA:: CSET_UTF8

.. data:: PAD_ZERO
.. data:: PAD_ONE
.. data:: PAD_BACKGROUND

.. data:: BKG_NO
.. data:: BKG_TEMP
.. data:: BKG_YES
