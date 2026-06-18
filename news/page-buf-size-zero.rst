Bug fixes
---------

* ``File(..., page_buf_size=0)`` now accepts an integer ``0`` and treats it the
  same as the string ``"0"``. Previously an integer ``0`` was silently ignored
  (the same as the ``None`` default), which also dropped any ``min_meta_keep``
  and ``min_raw_keep`` values passed alongside it. As a consequence, a
  nonsensical empty string ``page_buf_size=""`` now raises ``ValueError``
  instead of being silently ignored as before.
