Bug fixes
---------

* Avoid reading unselected variable-length fields from compound datasets,
  preventing memory growth when reading only other fields.
