Announcing HDF5 for Python (h5py) 2.5.0
========================================

The h5py team is happy to announce the availability of h5py 2.5.0.

This release introduces experimental support for the highly-anticipated
"Single Writer Multiple Reader" (SWMR) feature in the upcoming HDF5 1.10
release.  SWMR allows sharing of a single HDF5 file between multiple processes
without the complexity of MPI or multiprocessing-based solutions.  

This is an experimental feature that should NOT be used in production code.
We are interested in getting feedback from the broader community with respect
to performance and the API design.

For more details, check out the h5py user guide:
http://docs.h5py.org/en/latest/swmr.html

SWMR support was contributed by Ulrik Pedersen.


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

* Experimental SWMR support
* Group and AttributeManager classes now inherit from the appropriate ABCs
* Fixed an issue with 64-bit float VLENS
* Cython warning cleanups related to "const"
* Entire code base ported to "six"; 2to3 removed from setup.py
  

Acknowledgements
----------------

This release incorporates changes from, among others:

* Ulrik Pedersen
* James Tocknell
* Will Parkin
* Antony Lee
* Peter H. Li
* Peter Colberg
* Ghislain Antony Vaillant


Where to get it
---------------

Downloads, documentation, and more are available at the h5py website:

http://www.h5py.org
