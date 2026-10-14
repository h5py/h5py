Bug fixes
---------

* Fixed a reference leak when reading variable-length data. Each converted
  element leaked one reference to the element dtype object, so repeated reads
  of a vlen dataset kept its dtype alive indefinitely.
