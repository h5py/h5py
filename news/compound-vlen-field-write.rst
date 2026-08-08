Bug fixes
---------

* Writing a single variable length field of a compound dataset, e.g.
  ``ds['a_string'] = 'some text'``, no longer raises ``TypeError: Cannot change
  data-type for array of references``. Variable length data is stored in object
  arrays, which NumPy will not reinterpret as a compound dtype, so the value is
  now copied into a one field array instead of being viewed as one.
