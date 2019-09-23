New features
------------

* Migrate all Cython code base to Cython3 syntax. 
	* The only noticeable change is in exception raising from cython which use bytes.
	* Massively use local imports everywhere as expected from Python3
* Use the libc cimport. Note that the numpy is left untouched, cleanup needed in numpy.pdx. 
* All the libhdf5 binding is now nogil-enabled but not used
* Use *emalloc* in the _conv module to gracefully fail when no more memory is available
