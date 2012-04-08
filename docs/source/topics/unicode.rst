
Unicode and byte strings
========================

Starting with h5py 2.0, Unicode support in h5py is much improved.  However,
there may be a few surprises for people using h5py on Python 2.X.

Overview
--------

Unicode is handled slightly differently in Python 2 and Python 3.  In Python
2, the default string ("str") type represents a sequence of bytes.  An 
additional Unicode type ("unicode") stores a sequence of Unicode code points.
You convert from "unicode" to "str" by specifying an encoding format, which
tells Python how to map the Unicode code points into a byte string.  The
opposite process, `decoding`, takes a sequence of bytes known to be in some
encoding format and translates them back into a sequence of code points.

The drawback in the Python 2 world is the "str" type was generally used for
both raw byte strings `and` to represent text data.  People had to keep track
of their encodings manually, and the net result was a lot of hard-to-find bugs
and odd behavior.  Keeping text data in Unicode strings and encoded data in 
byte strings required great discipline and was prone to error, since most
routines in Python happily accepted byte strings in text-oriented roles.
With the move to Python 3, it was declared that from now on
the default string type in Python would be a Unicode string object, and raw
bytes would be represented by a distinct type ("bytes").  Bytes objects would
no longer be permitted for most text-oriented roles in Python.  Encoding and
decoding would have to be explicit and mixing types would not be allowed.

How h5py handles Unicode for object names
-----------------------------------------
At the C level in HDF5, all object names are represented by byte strings.  The
HDF Group has chosen the UTF-8 encoding as the preferred way to map Unicode
text into byte strings for this purpose.  There are two flavors of string
(character sets) defined in the HDF5 C library, "H5T_CSET_ASCII", which has 
come to denote either ASCII data or data in an unknown encoding, and 
"H5T_CSET_UTF_8", which indicates data in the UTF-8 encoding.

At the low level (modules ``h5py.h5*``), all routines accept only byte strings.
That means "str" on Python 2 or "bytes" on Python 3.

Starting with version 2.0, all "high-level" routines (like ``create_dataset``)
will accept either Unicode or byte strings.

    * If you provide a Unicode string   (type "unicode" on Python 2 or "str"
      on Python 3), h5py will encode it down to UTF-8 and use that as the name
      of the object in the file.  The type of the string will be recorded as
      H5T_CSET_UTF8 where appropriate
    * If you provide a byte string (type "str" on Python 3 or "bytes" on
      Python 3), h5py will use the string directly, and record its type as
      H5T_CSET_ASCII

When object names are retrieved from a file (for example, by ``Group.keys()``),
they will now be returned as Unicode strings if possible:

    * H5py will attempt to decode the object's name from UTF-8 into Unicode.
      If successful, a Unicode string ("unicode" on Python 2 and "str" on
      Python 3) will be returned.
    * If the decoding fails, for example, if the name is in some encoding
      other than UTF-8, a byte string ("str" on Python 3 or "bytes" on Python
      3) will be returned.

File names
----------

File names can be specified in either Unicode or byte strings when opening or
creating a file.  If Unicode is given, h5py will attempt to encode the string
using the file system endcoding.  If a byte string is given, h5py will pass it
on to the operating system as-is.

When retrieving a file name (via File.filename), h5py will attempt to decode it
and return a Unicode string.  If this fails, h5py will return a byte string.

Variable-length strings
-----------------------

Presently, variable-length Unicode strings are not supported.  When creating
a vlen string data type the only allowed types are "str" (Python 2) or "bytes"
(Python 3)::

    >>> dtype = h5py.special_dtype(vlen=bytes)  # Python 3 example

Fixed-length strings
--------------------

The NumPy "fixed-length" Unicode type (typecode "U") is not supported.  HDF5
does not define a fixed-width "wide character" type, only the variable-width
UTF-8 encoding.
