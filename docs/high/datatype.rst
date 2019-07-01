.. _datatype:

Datatypes
=========

Usually, :class:`Datatype` objects are created for you on the fly and you do not
have to worry about them. However, in certain cases, namely when no HDF5
equivalent exists for a given :class:`dtype`, you must register the :class:`dtype`
manually for use with h5py.::

    arr = np.array([np.datetime64('2019-06-30')])
    h5py.register_dtype(arr.dtype)
    dset = f.create_dataset("datetimes", data=arr)

Reference
=========

.. class:: Datatype(identifier)

    Datatype objects are typically created via :func:`register_dtype` function,
    or by retrieving existing datatype from a :class:`Dataset`. This class is a
    thin wrapper interface around :class:`dtype` and HDF5 types.

.. function:: register_dtype(dtype dt_in, bytes tag=None)

    Register a NumPy dtype for use with h5py. Types registered in this way
    will be stored as a custom opaque type, with a special tag to map it to
    the corresponding NumPy type.

    Opaque types with this tag will be mapped to NumPy types in the same way.

    The default tag is generated via the code:
    ``b"NUMPY:" + dt_in.descr[0][1].encode()``.

.. function:: deregister_dtype(object obj)

    Deregister a dtype/tag from the NumPy-tag mapping, along with the.
    corresponding tag/dtype.
