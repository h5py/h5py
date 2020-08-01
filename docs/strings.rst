.. _strings:

Strings in HDF5
===============

.. note::

   The rules around reading & writing string data were redesigned for h5py
   3.0. Refer to `the h5py 2.10 docs <https://docs.h5py.org/en/2.10.0/strings.html>`__
   for how to store strings in older versions.

Reading strings
---------------

String data in HDF5 datasets is read as bytes by default: ``bytes`` objects
for variable-length strings, or numpy bytes arrays (``'S'`` dtypes) for
fixed-length strings. Use :meth:`.Dataset.asstr` to retrieve ``str`` objects.

Variable-length strings in attributes are read as ``str`` objects. These are
decoded as UTF-8 with surrogate escaping for unrecognised bytes.

Storing strings
---------------

When creating a new dataset or attribute, Python ``str`` or ``bytes`` objects
will be treated as variable-length strings, marked as UTF-8 and ASCII respectively.
Numpy bytes arrays (``'S'`` dtypes) make fixed-length strings.
You can use :func:`.string_dtype` to explictly specify any HDF5 string datatype.

When writing data to an existing dataset or attribute, data passed as bytes is
written without checking the encoding. Data passed as Python ``str`` objects
is encoded as either ASCII or UTF-8, based on the HDF5 datatype.
In either case, null bytes (``'\x00'``) in the data will cause an error.

.. warning::

   Fixed-length string datasets will silently truncate longer strings which
   are written to them. Numpy byte string arrays do the same thing.

   Fixed-length strings in HDF5 hold a set number of bytes.
   It may take multiple bytes to store one character.

What about NumPy's ``U`` type?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NumPy also has a Unicode type, a UTF-32 fixed-width format (4-byte characters).
HDF5 has no support for wide characters.  Rather than trying to hack around
this and "pretend" to support it, h5py will raise an error if you try to store
data of this type.

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

Object names
------------

Unicode strings are used exclusively for object names in the file::

    >>> f.name
    '/'

You can supply either byte or unicode strings (on both Python 2 and Python 3)
when creating or retrieving objects. If a byte string is supplied,
it will be used as-is; Unicode strings will be encoded down to UTF-8.

In the file, h5py uses the most-compatible representation; H5T_CSET_ASCII for
characters in the ASCII range; H5T_CSET_UTF8 otherwise.

    >>> grp = f.create_dataset(b"name")
    >>> grp2 = f.create_dataset("name2")
