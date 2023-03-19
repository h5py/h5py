New features
------------

* Low-level ``h5p.set_elink_acc_flags`` function that allows to set the external link traversal file access flag in a link access property list.
* Low-level ``h5p.get_elink_acc_flags`` function that retrieves the external link traversal file access flag from the specified link access property list.

Deprecations
------------

* <news item>

Exposing HDF5 functions
-----------------------

* ``dlapl`` and ``dlcpl`` are accessible from the top level module to ease customisation of  default link access/creation property lists.

Bug fixes
---------

* ``Group.__contains__`` and ``Group.get`` now use the default link access property list systematically.

Building h5py
-------------

* <news item>

Development
-----------

* <news item>
