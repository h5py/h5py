Low-Level API
=============

This documentation mostly describes the h5py high-level API, which offers the
main features of HDF5 in an interface modelled on dictionaries and NumPy arrays.
h5py also provides a low-level API, which more closely follows the HDF5 C API.

.. seealso::

   - `h5py Low-Level API Reference <https://api.h5py.org/>`_
   - `HDF5 C/Fortran Reference Manual <https://confluence.hdfgroup.org/display/HDF5/Core+Library>`_

You can easily switch between the two levels in your code:

- **To the low-level**: High-level :class:`.File`, :class:`.Group` and
  :class:`.Dataset` objects all have a ``.id`` attribute exposing the
  corresponding low-level objects—:class:`~low:h5py.h5f.FileID`,
  :class:`~low:h5py.h5g.GroupID` and :class:`~low:h5py.h5d.DatasetID`::

      dsid = dset.id
      dsid.get_offset()  # Low-level method

  Although there is no high-level object for a single attribute,
  :meth:`.AttributeManager.get_id` will get the low-level
  :class:`~low:h5py.h5a.AttrID` object::

      aid = dset.attrs.get_id('timestamp')
      aid.get_storage_size()  # Low-level method

- **To the high-level**: Low-level :class:`~low:h5py.h5f.FileID`,
  :class:`~low:h5py.h5g.GroupID` and :class:`~low:h5py.h5d.DatasetID` objects
  can be passed to the constructors of :class:`.File`, :class:`.Group` and
  :class:`.Dataset`, respectively.
