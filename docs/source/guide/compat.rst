*******************
Compatibility Notes
*******************

File compatibility
==================

Types
-----

H5py now supports all native HDF5 types with the exception of generic VLENs
and time datatypes.  When a type cannot be understood, TypeError is raised
during conversion.

Low-level protocol only
-----------------------

The files h5py writes are "plain-vanilla" HDF5.  Higher-level protocols
(such as those which specify a mandatory series of attributes attached to
a dataset) are not used by h5py directly.  This allows h5py to achieve a
high degree of compatibility, at the cost of leaving higher level
protocols to the user.

LZF filter
----------

H5py includes a new high-speed compresion filter called LZF.  While a
standalone version of this filter (written in C) is available, it is unlikely
to be installed in many non-Python environments.  Use LZF with this in mind.

SZIP filter
-----------

HDF5 includes a compression filter called SZIP as part of the standard
distribution.  This filter contains patented software and consequently h5py
is unable to include the compression element in our binary releases, although
decompression of existing SZIP data is possible

HDF5 1.6/1.8 compatibility
--------------------------

The files written by HDF5 1.6 and 1.8 are almost always interoperable, unless
(in the case of a file written on 1.8) new features are used.  However, there
are reports of third-party software (certain older versions of Matlab and
IDL) which crash when reading a 1.8 file, even without new features.

Third-party applications
========================

PyTables
--------

H5py and PyTables seem to coexist well, even in the same thread.  If you
notice any problems, please let us know

Applications which use the HDF5 error system
--------------------------------------------

In order to function properly, h5py has to install a custom error handler.  It's
possible that a third-party program could be called from Python which does
not like this, either installing its own error handler over h5py's (thereby
breaking h5py), or having its handler replaced.

In order to address this, the h5py error handler is only installed (1) in the
main thread, and (2) in any thread which uses the high-level interface.
Additionally, it can be deactivated manually by calling 
h5py.h5e.unregister_thread() at any time.  If you encounter compatibility
problems with third-party HDF5 client code, please let us know.



