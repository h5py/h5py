Announcing HDF5 for Python (h5py) 2.3.1
=======================================

The h5py team is happy to announce the availability of h5py 2.3.1.  This is
a bugfix release which fixes a build issue when using the most recent
version of HDF5 (1.8.13), and some issues with Travis-CI.

What's h5py?
------------

The h5py package is a Pythonic interface to the HDF5 binary data format.

It lets you store huge amounts of numerical data, and easily manipulate
that data from NumPy. For example, you can slice into multi-terabyte
datasets stored on disk, as if they were real NumPy arrays. Thousands of
datasets can be stored in a single file, categorized and tagged however
you want.

Changes
-------

* Removed lingering references to the MPIPOSIX driver, which was removed
  in HDF5 1.8.13
* Resolved a build issue with Travis-CI, which caused good pull requests
  to be wrongly marked as failed.

Where to get it
---------------

Downloads, documentation, and more are available at the h5py website:

http://www.h5py.org
