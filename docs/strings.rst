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
for variable-length strings, or NumPy bytes arrays (``'S'`` dtypes) for
fixed-length strings. On NumPy >=2.0, use :meth:`astype('T')<.astype>`
to read into an array of native variable-width NumPy strings.
If you need backwards compatibility with NumPy 1.x or h5py <3.14, you should instead
call :meth:`.asstr` to retrieve an array of ``str`` objects.

Variable-length strings in attributes are read as ``str`` objects. These are
decoded as UTF-8 with surrogate escaping for unrecognised bytes. Fixed-length
strings are read as NumPy bytes arrays, the same as for datasets.

Storing strings
---------------

When creating a new dataset or attribute, Python ``str`` or ``bytes`` objects
will be treated as variable-length strings, marked as UTF-8 and ASCII respectively.
NumPy bytes arrays (``'S'`` dtypes) make fixed-length strings.
NumPy NpyStrings arrays (``StringDType`` or ``'T'`` dtypes) make native variable-length
strings.

You can use :func:`.string_dtype` to explicitly specify any HDF5 string datatype,
as shown in the examples below::

    string_data = ["varying", "sizes", "of", "strings"]

    # Variable length strings (implicit)
    f['vlen_strings1'] = string_data

    # Variable length strings (explicit)
    ds = f.create_dataset('vlen_strings2', shape=4, dtype=h5py.string_dtype())
    ds[:] = string_data

    # Fixed length strings (implicit) - longer strings are truncated
    f['fixed_strings1'] = np.array(string_data, dtype='S6')

    # Fixed length strings (explicit) - longer strings are truncated
    ds = f.create_dataset('fixed_strings2', shape=4, dtype=h5py.string_dtype(length=6))
    ds[:] = string_data

When writing data to an existing dataset or attribute, data passed as bytes is
written without checking the encoding. Data passed as Python ``str`` objects
is encoded as either ASCII or UTF-8, based on the HDF5 datatype.
In either case, null bytes (``'\x00'``) in the data will cause an error.
Data passed as NpyStrings is always encoded as UTF-8.

.. warning::

   Fixed-length string datasets will silently truncate longer strings which
   are written to them. NumPy byte string arrays do the same thing.

   Fixed-length strings in HDF5 hold a set number of bytes.
   It may take multiple bytes to store one character.

.. _npystrings:

NumPy variable-width strings
----------------------------

.. versionadded:: 3.14

Starting with NumPy 2.0 and h5py 3.14, you can also use ``np.dtypes.StringDType()``,
or ``np.dtype('T')`` for short, to specify native NumPy variable-width string dtypes,
a.k.a. NpyStrings.

However, note that when you open a dataset that you created as a StringDType, its dtype
will be object dtype with ``bytes`` elements. Likewise, :meth:`.create_dataset`
with parameter ``dtype='T'``, or with ``data=`` set to a NumPy array with StringDType,
will create a dataset with object dtype.
This is because the two data types are identical on the HDF5 file; this design is to
ensure that NumPy 1.x and 2.x behave in the same way unless the user explicitly requests
native strings.

In order to efficiently write NpyStrings to a dataset, simply assign a NumPy array with
StringDType to it, either through ``__setitem__`` or :meth:`.create_dataset`.
To read NpyStrings out of any string-type dataset, use
:meth:`astype('T')<.astype>`. In both cases, there will be no
intermediate conversion to object-type array, even if the dataset's dtype is object::

    # These three syntaxes are equivalent:
    >>> ds = f.create_dataset('x', shape=(2, ), dtype=h5py.string_dtype())
    >>> ds = f.create_dataset('x', shape=(2, ), dtype=np.dtypes.StringDType())
    >>> ds = f.create_dataset('x', shape=(2, ), dtype='T')

    # They all return a dataset with object dtype:
    >>> ds
    <HDF5 dataset "x": shape (2,), type "|O">

    # When you open an already existing dataset, you also get object dtype:
    >>> f['x']
    <HDF5 dataset "x": shape (2,), type "|O">

    # You can write a NumPy array with StringDType to it:
    >>> ds[:] = np.asarray(['hello', 'world'], dtype='T')

    # Reading back, you will get an object-type array of bytes by default:
    >>> ds[:]
    array([b'hello', b'world'], dtype=object)

    # To read it as a native NumPy variable-width string array, use .astype('T'):
    >>> ds.astype('T')[:]
    array(['hello', 'world'], dtype=StringDType())

    # The above is much faster than converting after reading:
    >>> ds[:].astype('T')  # Slower; don't do this!
    array(['hello', 'world'], dtype=StringDType())


What about NumPy's ``U`` type?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NumPy also has a Unicode fixed-width type, a UTF-32 fixed-width format
(4-byte characters). HDF5 has no support for wide characters.
Rather than trying to hack around this and "pretend" to support it,
h5py will raise an error if you try to store data of this type.

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
    >>> binary_blob = out.tobytes()

Object names
------------

Unicode strings are used exclusively for object names in the file::

    >>> f.name
    '/'

You can supply either byte or unicode strings
when creating or retrieving objects. If a byte string is supplied,
it will be used as-is; Unicode strings will be encoded as UTF-8.

In the file, h5py uses the most-compatible representation; H5T_CSET_ASCII for
characters in the ASCII range; H5T_CSET_UTF8 otherwise.

    >>> grp = f.create_dataset(b"name")
    >>> grp2 = f.create_dataset("name2")

.. _str_encodings:

Encodings
---------

HDF5 supports two string encodings: ASCII and UTF-8.
We recommend using UTF-8 when creating HDF5 files, and this is what h5py does
by default with Python ``str`` objects.
If you need to write ASCII for compatibility reasons, you should ensure you only
write pure ASCII characters (this can be done by
``your_string.encode("ascii")``), as otherwise your text may turn into
`mojibake <https://en.wikipedia.org/wiki/Mojibake>`_.
You can use :func:`.string_dtype` to specify the encoding for string data.

.. seealso::

   `Joel Spolsky's introduction to Unicode & character sets <https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/>`_
     If this section looks like gibberish, try this.

For reading, as long as the encoding metadata is correct, the defaults for
:meth:`.Dataset.asstr` will always work.
However, HDF5 does not enforce the string encoding, and there are files where
the encoding metadata doesn't match what's really stored.
Most commonly, data marked as ASCII may be in one of the many "Extended ASCII"
encodings such as Latin-1. If you know what encoding your data is in,
you can specify this using :meth:`.Dataset.asstr`. If you have data
in an unknown encoding, you can also use any of the `builtin python error
handlers <https://docs.python.org/3/library/codecs.html#error-handlers>`_.

Variable-length strings in attributes are read as ``str`` objects, decoded as
UTF-8 with the ``'surrogateescape'`` error handler. If an attribute is
incorrectly encoded, you'll see 'surrogate' characters such as ``'\udcb1'``
when reading it::

    >>> s = "2.0±0.1"
    >>> f.attrs["string_good"] = s  # Good - h5py uses UTF-8
    >>> f.attrs["string_bad"] = s.encode("latin-1")  # Bad!
    >>> f.attrs["string_bad"]
    '2.0\udcb10.1'

To recover the original string, you'll need to *encode* it with UTF-8,
and then decode it with the correct encoding::

    >>> f.attrs["string_bad"].encode('utf-8', 'surrogateescape').decode('latin-1')
    '2.0±0.1'

Fixed length strings are different; h5py doesn't try to decode them::

    >>> s = "2.0±0.1"
    >>> utf8_type = h5py.string_dtype('utf-8', 30)
    >>> ascii_type = h5py.string_dtype('ascii', 30)
    >>> f.attrs["fixed_good"] = np.array(s.encode("utf-8"), dtype=utf8_type)
    >>> f.attrs["fixed_bad"] = np.array(s.encode("latin-1"), dtype=ascii_type)
    >>> f.attrs["fixed_bad"]
    b'2.0\xb10.1'
    >>> f.attrs["fixed_bad"].decode("utf-8")
    Traceback (most recent call last):
      File "<input>", line 1, in <module>
        f.attrs["fixed_bad"].decode("utf-8")
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb1 in position 3: invalid start byte
    >>> f.attrs["fixed_bad"].decode("latin-1")
    '2.0±0.1'

As we get bytes back, we only need to decode them with the correct encoding.
