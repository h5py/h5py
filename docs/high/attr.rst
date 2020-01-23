.. currentmodule:: h5py
.. _attributes:


Attributes
==========

Attributes are a critical part of what makes HDF5 a "self-describing"
format.  They are small named pieces of data attached directly to
:class:`Group` and :class:`Dataset` objects.  This is the official way to
store metadata in HDF5.

Each Group or Dataset has a small proxy object attached to it, at
``<obj>.attrs``.  Attributes have the following properties:

- They may be created from any scalar or NumPy array
- Each attribute should be small (generally < 64k)
- There is no partial I/O (i.e. slicing); the entire attribute must be read.

The ``.attrs`` proxy objects are of class :class:`AttributeManager`, below.
This class supports a dictionary-style interface.

By default, attributes are iterated in alphanumeric order.  However,
if group or dataset is created with ``track_order=True``, the
attribute insertion order is remembered (tracked) in HDF5 file, and
iteration uses that order.  The latter is consistent with Python 3.7+
dictionaries.

The default ``track_order`` for all new groups and datasets can be
specified globally with ``h5.get_config().track_order``.


Reference
---------

.. class:: AttributeManager(parent)

    AttributeManager objects are created directly by h5py.  You should
    access instances by ``group.attrs`` or ``dataset.attrs``, not by manually
    creating them.

    .. method:: __iter__()

        Get an iterator over attribute names.

    .. method:: __contains__(name)

        Determine if attribute `name` is attached to this object.

    .. method:: __getitem__(name)

        Retrieve an attribute.

    .. method:: __setitem__(name, val)

        Create an attribute, overwriting any existing attribute.  The type
        and shape of the attribute are determined automatically by h5py.

    .. method:: __delitem__(name)

        Delete an attribute.  KeyError if it doesn't exist.

    .. method:: keys()

        Get the names of all attributes attached to this object.

        :return: set-like object.

    .. method:: values()

        Get the values of all attributes attached to this object.

        :return: collection or bag-like object.

    .. method:: items()

        Get ``(name, value)`` tuples for all attributes attached to this object.

        :return: collection or set-like object.

    .. method:: get(name, default=None)

        Retrieve `name`, or `default` if no such attribute exists.

    .. method:: get_id(name)

       Get the low-level :class:`AttrID <low:h5py.h5a.AttrID>` for the named
       attribute.

    .. method:: create(name, data, shape=None, dtype=None)

        Create a new attribute, with control over the shape and type.  Any
        existing attribute will be overwritten.

        :param name:    Name of the new attribute
        :type name:     String

        :param data:    Value of the attribute; will be put through
                        ``numpy.array(data)``.

        :param shape:   Shape of the attribute.  Overrides ``data.shape`` if
                        both are given, in which case the total number of
                        points must be unchanged.
        :type shape:    Tuple

        :param dtype:   Data type for the attribute.  Overrides ``data.dtype``
                        if both are given.
        :type dtype:    NumPy dtype


    .. method:: modify(name, value)

        Change the value of an attribute while preserving its type and shape.
        Unlike :meth:`AttributeManager.__setitem__`, if the attribute already
        exists, only its value will be changed.  This can be useful for
        interacting with externally generated files, where the type and shape
        must not be altered.

        If the attribute doesn't exist, it will be created with a default
        shape and type.

        :param name:    Name of attribute to modify.
        :type name:     String

        :param value:   New value.  Will be put through ``numpy.array(value)``.
