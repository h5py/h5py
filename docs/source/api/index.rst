
.. _api_ref:

#######################
Low-level API reference
#######################

While the :ref:`high-level component <user_guide>` provides a friendly,
NumPy-like interface
to HDF5, the major accomplishment of h5py is the low-level interface.  This
is the most powerful and flexible way to access HDF5, and is as close as you
can get the the library without having to actually write C code.

The goal of this layer is to provide access to (almost) the entire HDF5 C API,
while retaining a Pythonic, object-oriented style.  Emphasis has been placed
on clean design and exhaustive documentation.  Programmers familiar with the
HDF5 C API should find themselves at home.

You'll probably also find the
`official HDF5 documentation <http://www.hdfgroup.com/HDF5>`_
useful as a guide
to how the library itself operates.  In particular, the HDF5 User Guide is
an excellent description of each major component.

Version of this documentation
-----------------------------

.. automodule:: h5py.version

Two major versions of HDF5 currently exist; the 1.6 and 1.8 series.  Rather
than force you to use either version, h5py can be compiled in one of two
"API compatibility levels".  When compiled against HDF5 1.6 (or 1.8, using the
``--api=16`` compile-time switch), only HDF5 1.6 routines and structures are
available.  When compiled against HDF5 1.8 (``--api=18``), many new features
are available.

Despite the major differences between these two versions of HDF5, the library
has been designed for compatibility; the new feature set is a strict superset of
the old.  Code written against 1.6 will continue to work with 1.8.

Keep in mind that since this documentation is generated against the most recent
(1.8) API version available, some routines, arguments, keywords and even
data structures documented here may not be available in earlier versions.
Wherever possible, this is documented.

Low-level API reference
-----------------------

.. toctree::
    :maxdepth: 1

    low/h5
    low/h5a
    low/h5d
    low/h5e
    low/h5f
    low/h5fd
    low/h5g
    low/h5i
    low/h5l
    low/h5o
    low/h5p
    low/h5r
    low/h5s
    low/h5t
    low/h5z

Library organization
--------------------

Modules
~~~~~~~

While HDF5 is a C library, and therefore uses on global namespace for all
functions and constants, their naming scheme is designed to partition the API
into modules of related code.  H5py uses this as a guide for the Python-side
module organization.  For example, the Python wrapping of the HDF5 function
``H5Dwrite`` is contained in module ``h5d``, while the Python equivalent of
``H5Aiterate`` is in module ``h5a``.

Identifier wrapping
~~~~~~~~~~~~~~~~~~~

No matter how complete, a library full of C functions is not very fun to use.
Additionally, since HDF5 identifiers are natively expressed as integers, their
lifespan must be manually tracked by the library user.  This quickly becomes
impossible for applications of even a moderate size; errors will lead to
resource leaks or (in the worst case) accidentally invalidating identifiers.

Rather than a straight C-API mapping, all HDF5 identifiers are presented as
Python extension types.  The root type :class:`ObjectID <h5py.h5.ObjectID>`
provides a container for an integer identifier, which allows Python reference
counting to manage the lifespan of the identifer.  When no more references
exist to the Python object, the HDF5 identifier is automatically closed.

    >>> from h5py import h5s
    >>> sid = h5s.create_simple( (2,3) )
    >>> sid
    67108866 [1] (U) SpaceID
    >>> sid.id
    67108866

A side benefit is that many HDF5 functions take an identifier as their first
argument.  These are naturally expressed as methods on an identifier object.
For example, the HDF5 function``H5Dwrite`` becomes the method
:meth:`h5py.h5d.DatasetID.write`.  Code using this technique is easier to
write and maintain.

    >>> sid.select_hyperslab((0,0),(2,2))
    >>> sid.get_select_bounds()
    ((0L, 0L), (1L, 1L))

State & Hashing
~~~~~~~~~~~~~~~

Since the ``h5py.h5*`` family of modules is intended to be a straightforward
interface to HDF5, almost all state information resides with the HDF5 library
itself.  A side effect of this is that the hash and equality operations on
ObjectID instances are determined by the status of the underlying HDF5 object.
For example, if two GroupID objects with different HDF5 integer identifiers
point to the same group, they will have identical hashes and compare equal.
Among other things, this means that you can reliably use identifiers as keys
in a dictionary::

    >>> from h5py import h5f, h5g
    >>> fid = h5f.open('foo.hdf5')
    >>> grp1 = h5g.open(fid, '/')
    >>> grp2 = h5g.open(fid, '/')
    >>> grp1.id == grp2.id
    False
    >>> grp1 == grp2
    True
    >>> hash(grp1) == hash(grp2)
    True
    >>> x = {grp1: "The root group"}
    >>> x[grp2]
    'The root group'

.. note::
    Hashing is restricted to file-resident objects, as there needs to be a
    unique way to identify the object.

Data Conversion
---------------

The natural numerical layer between h5py and the Python user is NumPy.  It
provides the mechanism to transfer large datasets between HDF5 and Python
analysis routines.  Additionally, its type infrastructure ("dtype" objects)
closely matches the HDF5 hierarchy of types.  With very few exceptions, there
is good mapping between NumPy dtypes and HDF5 basic types.

The actual conversion between datatypes is performed by the optimised routines
inside the HDF5 library; all h5py does is provide the mapping between NumPy
and HDF5 type objects.  Because the HDF5 typing system is more comprehensive
than the NumPy system, this is an asymmetrical process. 

Translating from an HDF5 datatype object to a dtype results in the closest
standard NumPy representation of the datatype::

    >>> from h5py import h5t
    >>> h5t.STD_I32LE
    50331712 [1] (L) TypeIntegerID int32
    >>> h5t.STD_I32LE.dtype 
    dtype('int32')

In the vast majority of cases the two datatypes will have exactly identical
binary layouts, but not always.  For example, an HDF5 integer can have
additional leading or trailing padding, which has no NumPy equivalent.  In
this case the dtype will capture the logical intent of the type (as a 32-bit
signed integer), but not its layout.

The reverse transformation (NumPy type to HDF5 type) is handled by a separate
function.  It's guaranteed to result in an exact, binary-compatible
representation::

    >>> tid = h5t.py_create('=u8')
    >>> tid
    50331956 [1] (U) TypeIntegerID uint64

The HDF5 library contains translation routines which can handle almost any
conversion between types of the same class, including odd precisions and
padding combinations.  This process is entirely transparent to the user.








