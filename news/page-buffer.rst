New features
------------

* Enable setting file space page size when creating new HDF5 files. A new named argument ``fs_page_size`` is added to ``File()`` class.
* Enable HDF5 page buffering, a low-level caching feature, that may improve overall I/O performance in some cases. Three new named arguments are added to ``File()`` class: ``page_buf_size``, ``min_meta_keep``, and ``min_raw_keep``.
* Get and reset HDF5 page buffering statistics. Available as the low-level API of the ``FileID`` class.

Exposing HDF5 functions
-----------------------

* ``H5Freset_page_buffering_stats``
* ``H5Fget_page_buffering_stats``
* ``H5Pset_file_space_page_size``
* ``H5Pget_file_space_page_size``
* ``H5Pset_page_buffer_size``
* ``H5Pget_page_buffer_size``
