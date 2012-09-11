README file for h5py version 2.1.0
==================================

Websites
--------

  * Docs, general info: h5py.alfven.org
  * Downloads, FAQ, bug tracker:  h5py.googlecode.com
  * Mailing list: h5py at googlegroups

Prerequisites
-------------

You need, at a minimum:

  * HDF5 1.8.3 or later
  * Python 2.6, 2.7 or 3.2
  * Any modern version of NumPy

Optionally:

  * Cython 0.13 or later, to build from Mercurial
  * If using Python 2.6, unittest2 is needed to run the tests

Installing from tarball
-----------------------

Run the following commands:

  * python setup.py build [--hdf5=/path/to/hdf5]
  * python setup.py test   # optional
  * [sudo] python setup.py install

Installing via easy_install
---------------------------

Run the following commands:
 
  * export HDF5_DIR=/path/to/hdf5   # optional
  * [sudo] easy_install h5py

Building from a Mercurial checkout (UNIX)
-----------------------------------------

We now use Mercurial to manage changes at Google Code.  Here's how to build
h5py from source:

  * Clone the project:
    $ hg clone http://h5py.googlecode.com/hg h5py

  * Generate the Cython files which talk to HDF5:
    $ cd h5py/h5py
    $ python api_gen.py

  * Build the project (this step also auto-compiles the .c files)
    $ cd ..
    $ python setup.py build

  * Run the unit tests (optional)
    $ python setup.py test

Reporting bugs
--------------

  * Bug reports are always welcome at h5py.googlecode.com.  Please don't be
    offended if it takes a while to respond to your report... we value user
    input and take all bugs seriously.

  * If you're not sure you have a bug, or want to ask any question at all
    about h5py or HDF5, post to the mailing list (h5py at Google Groups).
    This list is read by the main developers of h5py, as well as the user
    community.  As a bonus, posting to the list means that other people with
    similar problems can find your question and the responses.

  * You're always free to email the author directly at [andrew dot collette
    at gmail dot com].

