Bug fixes
---------

* Writing a subclass of ``str`` or ``bytes``, such as an ``enum.StrEnum``
  member, as an attribute or dataset no longer raises ``TypeError: No conversion
  path for dtype: dtype('<U...')``. It is stored as a variable length string, like
  a plain ``str`` or ``bytes``. NumPy's ``np.str_`` and ``np.bytes_`` scalars are
  unaffected.
