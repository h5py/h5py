New features
------------

* New :meth:`h5py.Group.is_external` method to check whether accessing a path
  would use other files, via an external link, external dataset storage or a
  virtual dataset with sources in another file. It only inspects metadata in
  the current file, without opening any other files, so it can be used to
  check files from untrusted sources before accessing objects in them.
