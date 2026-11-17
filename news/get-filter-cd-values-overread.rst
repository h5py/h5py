Bug fixes
---------

* ``h5p.PropDCID.get_filter`` and ``get_filter_by_id`` no longer read past
  their 256-element ``cd_values`` buffer when a filter stores more than 256
  client data values. HDF5 reports the full count but fills only 256 entries,
  so the result is now clamped to the values actually returned instead of
  over-reading the stack buffer (or raising ``AssertionError`` with assertions
  enabled). Fixes :issue:`927`.
