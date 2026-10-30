Bug fixes
---------

* Fixed a heap buffer overflow when writing variable-length data whose
  elements are multidimensional arrays, or arrays of a different dtype from
  the first element. The conversion buffer was sized for a 1D array of the
  first element's dtype, so the whole array was copied past its end. Such
  elements now raise ``TypeError``.
