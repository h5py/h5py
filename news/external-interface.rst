New features
------------

* The ``external`` argument of :meth:`Group.create_dataset`, which argument
  specifies any external storage for the dataset, accepts more types
  (:issue:`1260`), as follows:

  * The top-level container may be any iterable, not only a list.
  * The names of external files may be not only ``str`` but also ``bytes`` or
    ``os.PathLike`` objects.
  * The offsets and sizes may be *numpy* integers as well as Python integers.

  See also the deprecation related to the ``external`` argument.

Deprecations
------------

* The ``external`` argument of :meth:`Group.create_dataset` no longer accepts
  the following forms (:issue:`1260`):

  * a list containing *name*, [*offset*, [*size*]];
  * a list containing *name1*, *name2*, …; and
  * a list containing tuples such as ``(name,)`` and ``(name, offset)`` that
    lack the offset or size.

  Furthermore, the *name*–*offset*–*size* triplets now must be a tuple rather
  than an arbitrary iterable.  See also the new feature related to the
  ``external`` argument.

Exposing HDF5 functions
-----------------------

* <news item>

Bug fixes
---------

* <news item>

Building h5py
-------------

* <news item>

Development
-----------

* <news item>
