
How do I... ?
=============


Files
-----

**For stuff like iteration, accessing and creating members, etc. see the section
on Groups below.  All File objects are also Group objects.**

Reference page is here.

Open & close
~~~~~~~~~~~~

Opening::

    >>> f = h5py.File('filename.hdf5')       # opens, or creates if it doesn't exist
    >>> f = h5py.File('filename.hdf5')       # same thing
    >>> f = h5py.File('filename.hdf5','r')   # readonly
    >>> f = h5py.File('filename.hdf5','r+')  # read/write
    >>> f = h5py.File('filename.hdf5','w')   # new file overwriting any existing file
    >>> f = h5py.File('filename.hdf5','w-')  # new file only if one doesn't exist

And to close:

    >>> f.close()

Note that all files must be explicitly closed; unlike Groups and Datasets,
``del f`` won't do the trick.


Open w/context manager
~~~~~~~~~~~~~~~~~~~~~~

File is closed at the end of the block, regardless of any exceptions::

    >>> with h5py.File('filename.hdf5') as f:
    ...     print f.keys()
    ...     do_other_stuff()

Get filename
~~~~~~~~~~~~

Note it's *not* ``f.name``; that's the name of the root group and is always ``/`` for File objects::

    >>> print f.filename

Get mode (readonly or r/w)
~~~~~~~~~~~~~~~~~~~~~~~~~~

This will always be one or 'r' or 'r+', even if you e.g. open with 'w'

    >>> print f.mode

Create a file with user block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    >>> f = h5py.File('name.hdf5', userblock_size=<size_in_bytes>)
    >>> print f.userblock_size

Block size must be a power of 2.  File must not already exist (use 'w' or 'w-'
mode).


Access a file's user block
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can open a user block by opening the HDF5 file as a regular Python file object.

1. The file must not be open in HDF5
2. Don't go past the user block region at the start of the file.  If in
   doubt, open the file with h5py and check the size (0 means no user block)

    >>> with h5py.File('name.hdf5','r') as f:   # Optional, but make sure you know the userblock size
    ...     size = f.userblock_size
    >>> f2 = file('name.hdf5','rb')
    >>> f2.write(b'x'*size)

Open a memory-resident file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the HDF5 CORE driver.

In-memory, dumped to 'name.hdf5' when closed::

    >>> f = h5py.File('name.hdf5', driver='core')

In-memory, discarded when closed::

    >>> f = h5py.File('name.hdf5', driver='core', backing_store=False)

Allocate memory in `size`-length chunks (if omitted, 64k)::

    >>> f = h5py.File('name.hdf5', driver='core', block_size=<size>)

Note that the file name is always required, even for purely in-memory files.

Datasets
--------

Read an entire dataset as a NumPy array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    >>> data = mydataset[...]

Read a scalar dataset
~~~~~~~~~~~~~~~~~~~~~

As a scalar value:

    >>> print mydataset[()]
    0.0

As a scalar NumPy array:

    >>> print mydataset[...]
    array(0.0, dtype=float32)













