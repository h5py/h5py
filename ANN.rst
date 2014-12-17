Announcing HDF5 for Python (h5py) 2.4.0
========================================

The h5py team is happy to announce the availability of h5py 2.4.0 (final).

What's h5py?
------------

The h5py package is a Pythonic interface to the HDF5 binary data format.

It lets you store huge amounts of numerical data, and easily manipulate
that data from NumPy. For example, you can slice into multi-terabyte
datasets stored on disk, as if they were real NumPy arrays. Thousands of
datasets can be stored in a single file, categorized and tagged however
you want.

Documentation is at:

http://docs.h5py.org

Changes
-------

This release incorporates a total re-write of the identifier management
system in h5py.  As part of this refactoring, the entire API is also now
protected by threading locks.  User-visible changes include:

* Files are now automatically closed when all objects within them
  are unreachable. Previously, if File.close() was not explicitly called,
  files would remain open and "leaks" were possible if the File object
  was lost.

* The entire API is now believed to be thread-safe (feedback welcome!).

* External links now work if the target file is already open.  Previously
  this was not possible because of a mismatch in the file close strengths.
  
* The options to setup.py have changed; a new top-level "configure"
  command handles options like --hdf5=/path/to/hdf5 and --mpi.  Setup.py 
  now works correctly under Python 3 when these options are used.
  
* Cython (0.17+) is now required when building from source.
  
* The minimum NumPy version is now 1.6.1.

* Various other enhancements and bug fixes
  
Acknowlegements
---------------

This release incorporates changes from, among others:

* Matthieu Brucher
* Laurence Hole
* John Tyree
* Pierre de Buyl
* Matthew Brett

Where to get it
---------------

Downloads, documentation, and more are available at the h5py website:

http://www.h5py.org
