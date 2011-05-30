What's new in h5py 1.4
======================

HDF5 for Python (h5py) 1.4 represents the first major refactoring of the h5py
codebase since the project's launch in 2008.  Many of the most important
changes are behind the scenes, and include changes to the way h5py interacts
with the HDF5 library and Python.  These changes have substantially
improved h5py's stability, and make it possible to use more modern versions
of HDF5 without compatibility concerns.  It is now also possible to use
h5py with Python 3.


Summary of compatibility changes
--------------------------------

* HDF5 1.8.3 through 1.8.7 now work correctly and are officially supported.

* Python 3.2 is officially supported by h5py!  Thanks especially to
  Darren Dale for getting this working.

* HDF5 1.6.X is no longer supported on any platform; following the release of
  1.6.10 some time ago, this branch is no longer maintained by The HDF Group.

* Python 2.6 or later is now required to run h5py.  This is a consequence of
  the numerous changes made to h5py for Python 3 compatibility.

* On Python 2.6, unittest2 is now required to run the test suite.


Character Encoding
------------------

As part of the port to Python 3, byte and Unicode strings are now strictly
treated within h5py.  Object names (names of groups, datasets, etc.) may be
given as either byte strings ("str" in 2.6/2.7, "bytes" in 3.2), or unicode
strings ("unicode" in 2.6/2.6, "str" in 3.2).  When Unicode strings are used,
they are encoded to UTF-8 for storage in the file, with the appropriate flags
set.  When byte strings are used, they are passed on to the file as-is, and
the default character-set flag (H5T_CSET_ASCII) is used.

When names are read (for example, by "obj.name" or during iteration), they
are decoded from UTF-8 in the file, and Unicode strings are returned.  This is
the case on all versions of Python, including 2.X.  If something goes wrong
during decoding of a name (for example, if there are non-UTF-8 byte sequences),
a byte string will be returned.


Top enhancements and bug fixes
------------------------------

* Fill values can now be specified when creating a dataset.  The fill time is
  H5D_FILL_TIME_IFSET for contiguous datasets, and H5D_FILL_TIME_ALLOC for
  chunked datasets.

* Slicing semantics for scalar datasets now matches NumPy's behavior; for a
  scalar dataset, dset[...] produces an ndarray with shape "()", while
  dset[()] produces a NumPy array scalar of the correct type.

* On Python 3, dictionary-style methods like Group.keys() and Group.values()
  return view-like objects instead of lists.

* Element retrival from datasets of compound type now properly returns an
  instance of numpy.void, instead of a tuple.

* Object and region references now work correctly in compound types.

* Zero-length dimensions for extendible axes are now allowed.

* H5py no longer attempts to auto-import ipython on startup.

* File format bounds can now be given when opening a high-level File object
  (keyword "libver").

* Group, Dataset and Datatype constructors can no longer be used to create or
  access resources.  Rather, each constructor lets you bind the new object
  to a corresponding low-level identifier (GroupID, DatasetID, etc).
  Creating objects in this fashion is now an official part of the API.

* Many, many other bug fixes.


Deprecations and removals
-------------------------

* The previously deprecated dict aliases for Group methods (listnames, etc)
  have been removed.

* The selections module has been removed.

* Module h5py.h5e has been removed.

* The H5Error exception class has been removed.  Native Python exceptions are
  now used exclusively.


Known issues
------------

* Threading support in 1.4-beta is still undergoing testing.

* There are reports of crashes related to storing object and region references.
  If this happens to you, please post on the mailing list or contact the h5py
  author directly.









