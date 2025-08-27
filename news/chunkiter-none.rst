New features
------------

* :meth:`h5py.Dataset.iter_chunks` accepts slice objects with the ``None`` value for ``slice.start`` and ``slice.stop`` attributes, or integers. Example: ``dset.iter_chunks((slice(None, None), 4))``. This is equivalent of ``dset[:,4]``.

Deprecations
------------

* <news item>

Exposing HDF5 functions
-----------------------

* <news item>

Bug fixes
---------


Building h5py
-------------

* <news item>

Development
-----------

* <news item>
