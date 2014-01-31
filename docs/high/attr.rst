.. _attributes:


HDF5 Attributes
===============

Groups and datasets can have small bits of named information attached to them.
This is the official way to store metadata in HDF5.  Each of these objects
has a small proxy object (:class:`AttributeManager`) attached to it as
``<obj>.attrs``.  Attributes have the following properties:

- They may be created from any scalar or NumPy array
- Each attribute should be small (generally < 64k)
- There is no partial I/O (i.e. slicing); the entire attribute must be read.

They support the same dictionary API as groups.

Reference
---------

auto



