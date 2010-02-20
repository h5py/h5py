.. _hlfile:

============
File Objects
============

Opening & creating files
------------------------

HDF5 files work generally like standard Python file objects.  They support
standard modes like r/w/a, and should be closed when they are no longer in
use.  However, there is obviously no concept of "text" vs "binary" mode.

    >>> f = h5py.File('myfile.hdf5','r')

The file name may be either ``str`` or ``unicode``. Valid modes are:

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
HDF5 file, optionally written out to disk when it is closed.  See the File
class documentation for an exhaustive list.


Reference
---------

In addition to the properties and methods defined here, File objects inherit
the full API of Group objects; in this case, the group in question is the
*root group* (/) of the file.

.. note::
    
    Please note that unlike Python file objects, and h5py.File objects from
    h5py 1.1, the attribute ``File.name`` does *not* refer to the file name
    on disk.  ``File.name`` gives the HDF5 name of the root group, "``/``". To
    access the on-disk name, use ``File.filename``.

.. autoclass:: h5py.File

    **File properties**

    .. autoattribute:: h5py.File.filename
    .. autoattribute:: h5py.File.mode
    .. autoattribute:: h5py.File.driver

    **File methods**

    .. automethod:: h5py.File.close
    .. automethod:: h5py.File.flush

    **Properties common to all HDF5 objects:**

    .. autoattribute:: h5py.File.file
    .. autoattribute:: h5py.File.parent
    .. autoattribute:: h5py.File.name
    .. autoattribute:: h5py.File.id
    .. autoattribute:: h5py.File.ref
    .. autoattribute:: h5py.File.attrs

