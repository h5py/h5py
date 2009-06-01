.. _attributes:

==========
Attributes
==========

Groups and datasets can have small bits of named information attached to them.
This is the official way to store metadata in HDF5.  Each of these objects
has a small proxy object (:class:`AttributeManager`) attached to it as
``<obj>.attrs``.  This dictionary-like object works like a :class:`Group`
object, with the following differences:

- Entries may only be scalars and NumPy arrays
- Each attribute must be small (recommended < 64k for HDF5 1.6)
- No partial I/O (i.e. slicing) is allowed for arrays

They support the same dictionary API as groups.

Reference
---------

.. class:: AttributeManager

    .. method:: __getitem__(name) -> NumPy scalar or ndarray

        Retrieve an attribute given a string name.

    .. method:: __setitem__(name, value)

        Set an attribute.  Value must be convertible to a NumPy scalar
        or array.

    .. method:: __delitem__(name)

        Delete an attribute.

    .. method:: create(name, data=None, shape=None, dtype=None)

        Create an attribute, optionally initializing it.

        name
            Name of the new attribute (required)

        data
            An array to initialize the attribute. 
            Required unless "shape" is given.

        shape
            Shape of the attribute.  Overrides data.shape if both are
            given.  The total number of points must be unchanged.

        dtype
            Data type of the attribute.  Overrides data.dtype if both
            are given.  Must be conversion-compatible with data.dtype.

    .. method:: modify(name, value)

        Change the value of an attribute while preserving its type.

        Differs from __setitem__ in that the type of an existing attribute
        is preserved.  Useful for interacting with externally generated files.

        If the attribute doesn't exist, it will be automatically created.

    .. method:: __len__

        Number of attributes

    .. method:: __iter__

        Yields the names of attributes

    .. method:: __contains__(name)

        See if the given attribute is present

    .. method:: keys

        Get a list of attribute names

    .. method:: iterkeys

        Get an iterator over attribute names

    .. method:: values

        Get a list with all attribute values

    .. method:: itervalues

        Get an iterator over attribute values

    .. method:: items

        Get an list of (name, value) pairs for all attributes.

    .. method:: iteritems

        Get an iterator over (name, value) pairs

    .. method:: get(name, default)

        Return the specified attribute, or default if it doesn't exist.

