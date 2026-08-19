Bug fixes
---------

* Importing h5py no longer warns that "h5py is running against HDF5 X when it
  was built against Y" when the two versions are ABI compatible (:pr:`2955`).
  From 2.0.0 onwards HDF5 guarantees that an application linked against X.Y.Z
  runs unmodified against any later X.A.B, so a build against e.g. HDF5 2.1.0
  running on 2.2.0 is now accepted silently. Downgrades, changes of major
  version, and any mismatch in the 1.x series (which predates that guarantee)
  still warn as before.
