==============================
Low-level (``h5py.h5*``) guide
==============================

This is a general overview of the lowest-level API in h5py, the layer that
calls into HDF5 directly.  A lot of effort has been put into making even this
component useful in a Python context.  It provides the most general interface
to HDF5, including the vast majority of the C library.

The definitive documentation for this level is available through docstrings.
`Auto-generated HTML documentation`__ based on these is available.
You'll probably also find the `official HDF5 documentation`__ useful as a guide
to how the library itself operates.  In particular, the HDF5 User Guide is
an excellent description of each major component.

__ http://h5py.alfven.org/docs
__ http://hdf.ncsa.uiuc.edu/HDF5/doc/index.html


Library organization
====================

Modules
-------

While HDF5 is a C library, and therefore uses on global namespace for all
functions and constants, their naming scheme is designed to partition the API
into modules of related code.  H5py uses this as a guide for the Python-side
module organization.  For example, the Python wrapping of the HDF5 function
``H5Dwrite`` is contained in module ``h5d``, while the Python equivalent of
``H5Aiterate`` is in module ``h5a``.

Identifier wrapping
-------------------

No matter how complete, a library full of C functions is not very fun to use.
Additionally, since HDF5 identifiers are natively handled as integers, their
lifespan must be manually tracked by the library user.  This quickly becomes
impossible for applications of even a moderate size; errors will lead to
resource leaks or (in the worst case) accidentally invalidating identifiers.

Rather than a straight C-API mapping, all HDF5 identifiers are presented as
Python extension types.  The root type ``h5.ObjectID`` provides a container
for an integer identifier, which allows Python reference counting to manage
the lifespan of the identifer.  When no more references exist to the Python
object, the HDF5 identifier is automatically closed.

A side benefit is that many HDF5 functions take an identifier as their first
argument.  They are naturally expressed as methods on an identifier object.
For example, the HDF5 function``H5Dwrite`` becomes the method
``h5d.DatasetID.write``.  Code using this technique is easier to write and
maintain.

State & Hashing
---------------

Since the ``h5py.h5*`` family of modules is intended to be a straightforward
interface to HDF5, almost all state information resides with the HDF5 library
itself.  A side effect of this is that the hash and equality operations on
ObjectID instances are determined by the status of the underlying HDF5 object.
For example, if two GroupID objects with different HDF5 integer identifiers
point to the same group, they will have identical hashes and compare equal.
Among other things, this means that if you can use
ObjectID/GroupID/DatasetID/etc. instances as keys in a dictionary.


Data Conversion
===============

The natural numerical layer between h5py and the Python user is NumPy.  It
provides the mechanism to transfer large datasets between HDF5 and Python
analysis routines.  Additionally, its type infrastructure ("dtype" objects)
closely matches the HDF5 hierarchy of types.  With very few exceptions, there
is good mapping between NumPy dtypes and HDF5 basic types.

The actual conversion between datatypes is performed by the optimised routines
inside the HDF5 library; all h5py does is provide the mapping between NumPy
and HDF5 type objects.  Because the HDF5 typing system is more comprehensive
than the NumPy system, this is an asymmetrical process. While translating
from ``NumPy => HDF5`` always results in a bit-for-bit identical description,
the reverse process ``HDF5 => NumPy`` cannot be guaranteed to result in an
exact description.  In the vast majority of cases, this does not matter; HDF5
can natively auto-convert a huge variety of representations.

API Versioning
==============

HDF5 recently went though a major release, in the form of version 1.8.0.
In addition to various stability improvements, it introduces a number of
new and changed functions.  Rather than force people to use a particular
version, h5py deals with this by specifying an "API compatibility" level.
In "1.6" mode, the extension can be compiled with either 1.6.X or 1.8.X, and
will function identically.  In this mode, only the functions from the 1.6.X
series are exposed.  In "1.8" mode, new features and function signatures from
HDF5 1.8.X are available.

Currently, while h5py can be built in both modes, not many 1.8.X features are
available.










