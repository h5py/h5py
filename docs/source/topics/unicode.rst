
Unicode and strings in h5py
===========================

Starting with version 2.0, as part of the work to support Python 3, h5py 
gained support for Unicode object and file names.  In version 2.1, support
was added for Unicode strings in datasets and attributes.

Historically, h5py has supported two kinds of strings: (1) fixed-length byte 
strings (NumPy kind "S") and (2) variable length byte strings (Python-2 "str" 
objects).  However, the "native" kind of string on Python 3 is unicode.
For example, if you create an array from a string, you get NumPy kind "U":

    >>> out = numpy.array("Hello")
    >>> out.dtype
    numpy.dtype("<U5")

Starting with h5py version 2.1, when running on Python 3 h5py will try to
return Unicode strings whenever possible.  Since there 

Object names
------------
On both Python 2 and Python 3, the names of objects in the file are natively
unicode.  You can create objects by providing either Unicode or byte strings:

    >>> dataset1 = myfile.create_dataset(b"Hello", shape(1,))  # stored as utf-8
    >>> dataset2 = myfile.create_dataset(u"Hello", shape(1,))  # raw bytes stored

When using Unicode, the string is encoded to UTF-8 (which is the encoding
blessed by the HDF Group).  When using byte strings, the bytes are stored
directly in the file, with no encoding.  In such cases, the HDF5 specification
officially only allows ASCII characters, although this is not enforced.

Reading the names of objects in the file, on both Python 2 and Python 3, will
produce a Unicode string, regardless of how the name was created::

    >>> dataset1.name
    u"Hello"
    >>> dataset2.name
    u"Hello"

If the string can't be decoded into Unicode (for example, if the file uses an
illegal non-UTF-8 encoding), a byte string will be returned.

Strings in datasets and attributes
----------------------------------

There are four kinds of strings in HDF5 which can be used to store data in
datasets and attributes:

1. Fixed-length ASCII strings
2. Fixed-length UTF-8 strings
3. Variable-length ASCII strings
4. Variable-length UTF-8 strings

On the Python side, there are also (conveniently) four kinds of strings:

1. Fixed-length byte strings (NumPy kind "S")
2. Fixed-length unicode strings (NumPy kind "U")
3. Variable-length byte strings (Py2 "str", Py3 "bytes")
4. Variable-length unicode strings (Py2 "unicode", Py3 "str")

Here's how to create a dtype for each kind of string:

>>> fixed_byte    = numpy.dtype("|S10")              # (1)
>>> fixed_unicode = numpy.dtype("<U10")              # (2)
>>> vlen_bytes    = h5py.special_dtype(vlen=bytes)   # (3)
>>> vlen_unicode  = h5py.special_dtype(vlen=unicode) # (4)

You have to use a special function for (3) and (4) because Numpy has no
native representation for variable-length strings.  The function
special_dtype creates an object (kind "O") dtype with some metadata, that
tells h5py that the array contains byte or unicode string objects.

Creating strings from data or dtypes
-------------------------------------

Here's what you get when you ask h5py to create a dataset by providing either
raw data or an explicit dtype.  These rules are the same for both Python 2 and
Python 3.

=========================== =========================== =====================================
Input data                  Dtype                       HDF5 type
=========================== =========================== =====================================
numpy.array(b"Hellothere")  numpy.dtype("|S10")         Fixed-length ASCII string (length 10)
numpy.array(u"Hellothere")  numpy.dtype("<U10")         Variable-length UTF-8 string
b"Hellothere"               special_dtype(vlen=bytes)   Variable-length ASCII string
u"Hellothere"               special_dtype(vlen=unicode) Variable-length UTF-8 string
=========================== =========================== =====================================

This example is written for Python 2, although the same mapping holds on
Python 3.  Keep in minds that on Python 3, you'll have to replace `vlen=unicode`
with `vlen=str`, since `str` is the name for unicode strings on Python 3.

You might notice that creating a string from a fixed-length NumPy unicode
string ("<U10" in this example) creates a variable-length string.  Why is this?
There's an "impedance mismatch" between how Unicode is represented in Numpy
and HDF5 (UTF-32 vs. UTF-8).  What it boils down to is that there's no safe
way to store a fixed-length Unicode string that originates in NumPy, so we
store it as a variable-length string.  Keep this behavior in mind, as you'll
get unicode string objects (Python "unicode") back when you read the dataset!

Reading strings (Python 2)
--------------------------

On Python 2, strings will generally be returned as byte strings unless they
are explicitly named in the file as unicode.  Here's the mapping:

=============================   ============================================================
HDF5 type                       Returned type
=============================   ============================================================
Fixed-length ASCII              NumPy fixed-length byte string (dtype "S")
Fixed-length UTF-8              NumPy fixed-length unicode string (dtype "U") :ref:`[note!] <uwarning>`
Variable-length ASCII           Python bytes object (dtype is special_dtype(vlen=bytes))
Variable-length UTF-8           Python unicode object (dtype is special_dtype(vlen=unicode))
=============================   ============================================================

.. _uwarning:

.. note::
    Be careful when writing data to a dataset whose type is reported as NumPy 
    kind "U".  Since HDF5 uses a
    byte-oriented approach to storing strings, the length of the string
    (e.g. the "10" in dtype "<U10") actually refers to the `number of bytes`
    available for storage, rather than the number of characters available.

    If you're using non-ascii characters, they may not all fit in the space
    available in the file.  H5py will do its best to warn you when this happens.

    Generally speaking, the *only* safe way to store Unicode data in HDF5
    is by using variable-length strings.

Reading strings (Python 3)
--------------------------

On Python 3, strings will generally be returned as unicode strings:

=============================   ============================================================
HDF5 type                       Returned type
=============================   ============================================================
Fixed-length ASCII              NumPy fixed-length unicode string (dtype "U") :ref:`[note!] <uwarning>`
Fixed-length UTF-8              NumPy fixed-length unicode string (dtype "U") :ref:`[note!] <uwarning>`
Variable-length ASCII           Python unicode object (dtype is special_dtype(vlen=unicode))
Variable-length UTF-8           Python unicode object (dtype is special_dtype(vlen=unicode))
=============================   ============================================================

How to always return byte strings
---------------------------------

Sometimes, strings are used to hold data that isn't ASCII or UTF-8 text.  Or,
perhaps you would like to avoid the overhead of using multiple bytes per
character to store data that you know is going to be in the ASCII range.

You can make h5py return byte strings regardless of the platform (Py2 or Py3)
or character set by using the ``byte_strings`` context manager on the global
h5py configuration object.  Suppose we have a dataset with fixed-length strings
of length 10.  On Python 3, for convenience h5py will report the type of the
dataset as a fixed-with Unicode type (*warning:* :ref:`see note above<uwarning>`):

    >>> print(mydataset.dtype)
    numpy.dtype("<U10")

By enabling ``byte_strings``, we can always access the raw data as byte strings:

    >>> config = h5py.get_config()
    >>> with config.byte_strings:
    ...     print mydataset.dtype
    numpy.dtype("|S10")

The translation table (on both Py2 and Py3) when reading with ``byte_strings``
enabled is just what you'd expect:

=============================   ============================================================
HDF5 type                       Returned type
=============================   ============================================================
Fixed-length ASCII              NumPy fixed-length byte string (dtype "S")
Fixed-length UTF-8              NumPy fixed-length byte string (dtype "S")
Variable-length ASCII           Python bytes object (dtype is special_dtype(vlen=bytes))
Variable-length UTF-8           Python bytes object (dtype is special_dtype(vlen=bytes))
=============================   ============================================================

File names
----------

File names can be specified in either Unicode or byte strings when opening or
creating a file.  If Unicode is given, h5py will attempt to encode the string
using the file system endcoding.  If a byte string is given, h5py will pass it
on to the operating system as-is.

When retrieving a file name (via File.filename), h5py will attempt to decode it
and return a Unicode string.  If this fails, h5py will return a byte string.

