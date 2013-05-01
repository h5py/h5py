Announcing HDF5 for Python (h5py) 2.1.2
=======================================

HDF5 for Python 2.1.2 is now available!  This is a bugfix release, and the
second in our rapid release program.

Our new home
============

With h5py 2.1.1, development moved over to GitHub: http://github.com/h5py/h5py.

We welcome bug reports and pull requests from anyone interested in contributing.

Releases will now be made every 4-6 weeks, in order to get bugfixes and new
features out to users quickly while still leaving time for testing.

* New main website: http://www.h5py.org
* Mailing list:     http://groups.google.com/group/h5py


What is h5py?
=============

The h5py package is a Pythonic interface to the HDF5 binary data format.

It lets you store huge amounts of numerical data, and easily manipulate that 
data from NumPy. For example, you can slice into multi-terabyte datasets 
stored on disk, as if they were real NumPy arrays. Thousands of datasets can 
be stored in a single file, categorized and tagged however you want.

H5py uses straightforward NumPy and Python metaphors, like dictionary and 
NumPy array syntax. For example, you can iterate over datasets in a file, or 
check out the .shape or .dtype attributes of datasets. You don't need to know 
anything special about HDF5 to get started.

In addition to the easy-to-use high level interface, h5py rests on a 
object-oriented Cython wrapping of the HDF5 C API. Almost anything you can do 
from C in HDF5, you can do from h5py.

Best of all, the files you create are in a widely-used standard binary format, 
which you can exchange with other people, including those who use programs 
like IDL and MATLAB.


What's new in 2.1.2?
====================

This is a bugfix release.  The most substantial changes were:

* Fixed a memory leak related to string handling in exceptions
  (Thanks again to Luke Campbell)

* Fixed a longstanding bug preventing the use of the array type as the
  top-level type in a dataset

* The auto-chunker no longer freezes for datatypes larger than the
  optimum chunk size

* ``h5py.__version__`` now provides the current version string

