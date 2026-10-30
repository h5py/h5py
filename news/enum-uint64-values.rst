Bug fixes
---------

* Enum types with an unsigned 64-bit base type can now have members with
  values of ``2**63`` or more. Creating one, e.g. with
  ``h5py.enum_dtype({'HIGH': 2**63}, basetype='uint64')``, used to raise
  ``OverflowError``, and reading such a type from a file gave the wrong values
  for those members.
