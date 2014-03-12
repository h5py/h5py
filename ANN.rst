Announcing HDF5 for Python (h5py) 2.3.0 BETA
============================================

The h5py team is happy to announce the availability of h5py 2.3.0 beta. This
beta release will be available for approximately two weeks.

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

This release introduces some important new features, including:

* Support for arbitrary vlen data
* Improved exception messages
* Improved setuptools support
* Multiple additions to the low-level API
* Improved support for MPI features
* Single-step build for HDF5 on Windows

A complete description of changes is available online:

http://docs.h5py.org/en/latest/whatsnew/2.3.html

Where to get it
---------------

Downloads, documentation, and more are available at the h5py website:

http://www.h5py.org

Acknowledgements
----------------

The h5py package relies on third-party testing and contributions.  For the
2.3 release, thanks especially to:

* Martin Teichmann
* Florian Rathgerber
* Pierre de Buyl
* Thomas Caswell
* Andy Salnikov
* Darren Dale
* Robert David Grant
* Toon Verstraelen
* Many others who contributed bug reports and testing