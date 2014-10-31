Announcing HDF5 for Python (h5py) 2.4.0 BETA
============================================

The h5py team is happy to announce the availability of h5py 2.4.0 (beta).

This beta version will be available for approximately two weeks.  Because
of the substantial number of changes to the code base, we welcome feedback,
particularly from MPI users.

Documentation for the beta is at:

http://docs.h5py.org/en/latest/


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
  

Where to get it
---------------

Downloads, documentation, and more are available at the h5py website:

http://www.h5py.org
