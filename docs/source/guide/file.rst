.. _hlfile:

============
File Objects
============

Opening & creating files
------------------------

HDF5 files work generally like standard Python file objects.  They support
standand modes like r/w/a, and should be closed when they are no longer in
use.  However, there is obviously no concept of "text" vs "binary" mode.

    >>> f = h5py.File('myfile.hdf5','r')

Valid modes are:

    ===  ================================================
     r   Readonly, file must exist
     r+  Read/write, file must exist
     w   Create file, truncate if exists
     w-  Create file, fail if exists
     a   Read/write if exists, create otherwise (default)
    ===  ================================================

File drivers
------------

HDF5 ships with a variety of different low-level drivers, which map the logical
HDF5 address space to different storage mechanisms.  You can specify which
driver you want to use when the file is opened::

    >>> f = h5py.File('myfile.hdf5', driver=<driver name>, <driver_kwds>)

For example, the HDF5 "core" driver can be used to create a purely in-memory
HDF5 file, optionally written out to disk when it is closed.  Currently
supported drivers are:

None
    Use the standard HDF5 driver appropriate for the current platform.
    On UNIX, this is the H5FD_SEC2 driver; on Windows, it is
    H5FD_WINDOWS.

'sec2'
    Unbuffered, optimized I/O using standard POSIX functions.

'stdio' 
    Buffered I/O using functions from stdio.h.

'core'
    Memory-map the entire file; all operations are performed in
    memory and written back out when the file is closed.  Keywords:

        backing_store  
            If True (default), save changes to a real file
            when closing.  If False, the file exists purely
            in memory and is discarded when closed.

        block_size     
            Increment (in bytes) by which memory is extended.
            Default is 1 megabyte (1024**2).

'family'
    Store the file on disk as a series of fixed-length chunks.  Useful
    if the file system doesn't allow large files.  Note: the filename
    you provide *must* contain a printf-style integer format code (e.g "%d"),
    which will be replaced by the file sequence number.  Keywords:

        memb_size
            Maximum file size (default is 2**31-1).

Reference
---------

In addition to the properties and methods defined here, File objects inherit
the full API of Group objects; in this case, the group in question is the
*root group* (/) of the file.

.. class:: File

    Represents an HDF5 file on disk, and provides access to the root
    group (``/``).

    See also :class:`Group`, of which this is a subclass.

    .. attribute:: filename

        HDF5 filename on disk.  This is a plain string (str) for ASCII names,
        Unicode otherwise.

    .. attribute:: mode

        Mode (``r``, ``w``, etc) used to open file

    .. attribute:: driver

        Driver ('sec2', 'stdio', etc.) used to open file

    .. method:: __init__(name, mode='a', driver=None, **driver_kwds)
        
        Open or create an HDF5 file.  See above for a summary of options.
        Argument *name* may be an ASCII or Unicode string.

    .. method:: close()

        Close the file.  As with Python files, it's good practice to call
        this when you're done.

    .. method:: flush()

        Ask the HDF5 library to flush its buffers for this file.

