Bug fixes
---------

* Fix broadcasting to :class:`.MultiBlockSlice` selections, which could write to
  unselected elements or raise an out-of-bounds error. This also fixes
  broadcasting with :meth:`.Dataset.read_direct` and :meth:`.Dataset.write_direct`.
