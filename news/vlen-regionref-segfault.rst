Bug fixes
---------

* Fixed a crash when reading a dataset whose datatype is a variable-length
  sequence of region references. The background buffer used while converting
  the references was allocated without being zeroed, and the converter
  released whatever pointer it found there. This affected every platform whose
  allocator does not hand back zeroed memory.
