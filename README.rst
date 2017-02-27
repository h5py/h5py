.. image:: https://travis-ci.org/h5py/h5py.png
   :target: https://travis-ci.org/h5py/h5py
.. image:: https://ci.appveyor.com/api/projects/status/h3iajp4d1myotprc/branch/master?svg=true
   :target: https://ci.appveyor.com/project/h5py/h5py/branch/master

HDF5 for Python
===============

Websites
--------

* Main website: http://www.h5py.org
* Source code: http://github.com/h5py/h5py
* Mailing list: https://groups.google.com/d/forum/h5py

For advanced installation options, see http://docs.h5py.org.

Prerequisites
-------------

You need, at a minimum:

* Python 2.6, 2.7, 3.2, 3.3, or 3.4
* NumPy 1.6.1 or later
* The "six" package for Python 2/3 compatibility

To build on UNIX:

* HDF5 1.8.4 or later (on Windows, HDF5 comes with h5py)
* Cython 0.19 or later
* If using Python 2.6, unittest2 is needed to run the tests

Installing on Windows
---------------------

Download an installer from http://www.h5py.org and run it.

Installing on UNIX
------------------

Via pip (recommended)::
 
   pip install h5py

From a release tarball or Git checkout::

   python setup.py build
   python setup.py test # optional
   [sudo] python setup.py install
   
Reporting bugs
--------------

Open a bug at http://github.com/h5py/h5py/issues.  For general questions, ask
on the list (https://groups.google.com/d/forum/h5py).
