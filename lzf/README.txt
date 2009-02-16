===============================
LZF filter for HDF5, revision 1
===============================

The LZF filter provides high-speed compression with acceptable compression
performance, resulting in much faster performance than DEFLATE, at the
cost of a slightly worse compression ratio. It's appropriate for large
datasets of low to moderate complexity, for which some compression is
much better than none, but for which the speed of DEFLATE is unacceptable.

Both HDF5 versions 1.6 and 1.8 are supported.


Using the filter from HDF5
--------------------------

There is exactly one new public function declared in lzf_filter.h, with
the following signature:

    int register_lzf(void)

Calling this will register the filter with the HDF5 library.  A non-negative
return value indicates success.  If the registration fails, an error is pushed
onto the current error stack and a negative value is returned.

It's strongly recommended to use the SHUFFLE filter with LZF, as it's
cheap, supported by all current versions of HDF5, and can significantly
improve the compression ratio.  An example C program ("example.c") is included
which demonstrates the proper use of the filter.


Compiling
---------

The filter consists of a single .c file and header, along with an embedded
version of the LZF compression library.  Since the filter is stateless, it's
recommended to statically link the entire thing into your program; for
example:

    $ gcc -O2 -lhdf5 lzf/*.c lzf_filter.c myprog.c -o myprog

It can also be built as a shared library, although you will have to install
the resulting library somewhere the runtime linker can find it:

    $ gcc -02 -lhdf5 -fPIC -shared lzf/*.c lzf_filter.c -o liblzf_filter.so

This filter has not been tested with C++ code.  As in these examples, using
option -O1 or higher is strongly recommended for increased performance.


Contact
-------

This filter is maintained as part of the HDF5 for Python (h5py) project.  The
goal of h5py is to provide access to the majority of the HDF5 C API and feature
set from Python.

* Downloads and bug tracker:        http://h5py.googlecode.com

* Main web site and documentation:  http://h5py.alfven.org

* Contact email:  h5py at alfven dot org






