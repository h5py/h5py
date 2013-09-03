.. image:: https://travis-ci.org/h5py/h5py.png
   :target: https://travis-ci.org/h5py/h5py

HDF5 for Python
===============

Websites
--------

* Main website: http://www.h5py.org
* Source code: http://github.com/h5py/h5py
* Mailing list: h5py at googlegroups

Prerequisites
-------------

You need, at a minimum:

* HDF5 1.8.3 or later on Linux (On Windows, h5py ships with HDF5)
* Python 2.6, 2.7, 3.2 or 3.3
* Any modern version of NumPy

Optionally:

* Cython 0.13 or later, to build from a git checkout
* If using Python 2.6, unittest2 is needed to run the tests

Installing from release tarball
-------------------------------

Run the following commands::

   python setup.py build [--hdf5=/path/to/hdf5]
   python setup.py test   # optional
   [sudo] python setup.py install

Installing via easy_install
---------------------------

Run the following commands::
 
   export HDF5_DIR=/path/to/hdf5   # optional
   [sudo] easy_install h5py

Building from a Git checkout (UNIX)
-----------------------------------------

We have switched development to GitHub.  Here's how to build
h5py from source:

1. Clone the project::
   
      git clone https://github.com/h5py/h5py.git

2. Build the project (this step also auto-compiles the .c files)::
  
      python setup.py build [--hdf5=/path/to/hdf5]

3. Run the unit tests (optional)::
  
      python setup.py test

If you add new functions to api_functions.txt, remember to run the script
api_gen.py to update the Cython interface.  See the developer guide at
http://www.h5py.org for more information.

Reporting bugs
--------------

* Bug reports are always welcome at the GitHub tracker.  Please don't be
  offended if it takes a while to respond to your report... we value user
  input and take all bugs seriously.

* If you're not sure you have a bug, or want to ask any question at all
  about h5py or HDF5, post to the mailing list (h5py at Google Groups).
  This list is read by the main developers of h5py, as well as the user
  community.  As a bonus, posting to the list means that other people with
  similar problems can find your question and the responses.

* You're always free to email the author directly at [andrew dot collette
  at gmail dot com].
