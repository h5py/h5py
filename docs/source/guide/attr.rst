.. _attributes:

==========
Attributes
==========

Groups and datasets can have small bits of named information attached to them.
This is the official way to store metadata in HDF5.  Each of these objects
has a small proxy object (:class:`AttributeManager`) attached to it as
``<obj>.attrs``.  Attributes have the following properties:

- They may be created from any scalar or NumPy array
- Each attribute must be small (recommended < 64k for HDF5 1.6)
- There is no partial I/O (i.e. slicing); the entire attribute must be read.

They support the same dictionary API as groups.

Reference
---------

.. autoclass:: h5py.AttributeManager

    .. automethod:: h5py.AttributeManager.__getitem__
    .. automethod:: h5py.AttributeManager.__setitem__
    .. automethod:: h5py.AttributeManager.__delitem__

    .. automethod:: h5py.AttributeManager.create
    .. automethod:: h5py.AttributeManager.modify

    **Inherited dictionary interface**

    .. automethod:: h5py.highlevel._DictCompat.keys
    .. automethod:: h5py.highlevel._DictCompat.values
    .. automethod:: h5py.highlevel._DictCompat.items

    .. automethod:: h5py.highlevel._DictCompat.iterkeys
    .. automethod:: h5py.highlevel._DictCompat.itervalues
    .. automethod:: h5py.highlevel._DictCompat.iteritems

    .. automethod:: h5py.highlevel._DictCompat.get



