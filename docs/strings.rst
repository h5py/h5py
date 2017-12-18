.. _strings:

Strings in HDF5
===============

The Most Important Thing
------------------------

If you remember nothing else, remember this:

    **All strings in HDF5 hold encoded text.**

You *can't* store arbitrary binary data in HDF5 strings.  Not only will this
break, it will break in odd, hard-to-discover ways that will leave
you confused and cursing.


.. _str_binary:

How to store raw binary data
----------------------------

If you have a non-text blob in a Python byte string (as opposed to ASCII or
UTF-8 encoded text, which is fine), you should wrap it in a ``void`` type for
storage. This will map to the HDF5 OPAQUE datatype, and will prevent your
blob from getting mangled by the string machinery.

Here's an example of how to store binary data in an attribute, and then
recover it::

    >>> binary_blob = b"Hello\x00Hello\x00"
    >>> dset.attrs["attribute_name"] = np.void(binary_blob)
    >>> out = dset.attrs["attribute_name"]
    >>> binary_blob = out.tostring()



How to store text strings
-------------------------

At the high-level interface, h5py exposes three kinds of strings.  Each maps
to a specific type within Python (but see :ref:`str_py3` below):

* Fixed-length ASCII (NumPy ``S`` type)
* Variable-length ASCII (Python 2 ``str``, Python 3 ``bytes``)
* Variable-length UTF-8 (Python 2 ``unicode``, Python 3 ``str``)

.. _str_py3:

Compatibility
^^^^^^^^^^^^^

If you want to write maximally-compatible files and don't want to read the
whole chapter:

* Use ``numpy.string_`` for scalar attributes
* Use the NumPy ``S`` dtype for datasets and array attributes


Fixed-length ASCII
^^^^^^^^^^^^^^^^^^

These are created when you use ``numpy.string_``:

    >>> dset.attrs["name"] = numpy.string_("Hello")

or the ``S`` dtype::

    >>> dset = f.create_dataset("string_ds", (100,), dtype="S10")

In the file, these map to fixed-width ASCII strings.  One byte per character
is used.  The representation is "null-padded", which is the internal
representation used by NumPy (and the only one which round-trips through HDF5).

Technically, these strings are supposed to store `only` ASCII-encoded text,
although in practice anything you can store in NumPy will round-trip.  But
for compatibility with other programs using HDF5 (IDL, MATLAB, etc.), you
should use ASCII only.

.. note::

    This is the most-compatible way to store a string.  Everything else
    can read it.

Variable-length ASCII
^^^^^^^^^^^^^^^^^^^^^

These are created when you assign a byte string to an attribute::

    >>> dset.attrs["attr"] = b"Hello"

or when you create a dataset with an explicit "bytes" vlen type::

    >>> dt = h5py.special_dtype(vlen=bytes)
    >>> dset = f.create_dataset("name", (100,), dtype=dt)

Note that they're `not` fully identical to Python byte strings.  You can
only store ASCII-encoded text, without NULL bytes::

    >>> dset.attrs["name"] = b"Hello\x00there"
    ValueError: VLEN strings do not support embedded NULLs

In the file, these are created as variable-length strings with character set
H5T_CSET_ASCII.


Variable-length UTF-8
^^^^^^^^^^^^^^^^^^^^^

These are created when you assign a ``unicode`` string to an attribute::

    >>> dset.attrs["name"] = u"Hello"

or if you create a dataset with an explicit ``unicode`` vlen type:

    >>> dt = h5py.special_dtype(vlen=unicode)
    >>> dset = f.create_dataset("name", (100,), dtype=dt)

They can store any character a Python unicode string can store, with the
exception of NULLs.  In the file these are created as variable-length strings
with character set H5T_CSET_UTF8.


Exceptions for Python 3
^^^^^^^^^^^^^^^^^^^^^^^

Most strings in the HDF5 world are stored in ASCII, which means they map to
byte strings.  But in Python 3, there's a strict separation between `data` and
`text`, which intentionally makes it painful to handle encoded strings
directly.

So, when reading or writing scalar string attributes, on Python 3 they will
`always` be returned as type ``str``, regardless of the underlying storage
mechanism.  The regular rules for writing apply; to get a fixed-width ASCII
string, use ``numpy.string_``, and to get a variable-length ASCII string, use
``bytes``.


What about NumPy's ``U`` type?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NumPy also has a Unicode type, a UTF-32 fixed-width format (4-byte characters).
HDF5 has no support for wide characters.  Rather than trying to hack around
this and "pretend" to support it, h5py will raise an error when attempting
to create datasets or attributes of this type.


Object names
------------

Unicode strings are used exclusively for object names in the file::

    >>> f.name
    u'/'

You can supply either byte or unicode strings (on both Python 2 and Python 3)
when creating or retrieving objects. If a byte string is supplied,
it will be used as-is; Unicode strings will be encoded down to UTF-8.

In the file, h5py uses the most-compatible representation; H5T_CSET_ASCII for
characters in the ASCII range; H5T_CSET_UTF8 otherwise.

    >>> grp = f.create_dataset(b"name")
    >>> grp2 = f.create_dataset(u"name2")
