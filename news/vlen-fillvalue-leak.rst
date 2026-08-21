Bug fixes
---------

* Fixed a memory leak when reading the fill value of a dataset with a
  variable-length string datatype. Each access to :attr:`Dataset.fillvalue`
  leaked the string buffer allocated by HDF5.
