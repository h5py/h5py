.. currentmodule:: h5py
.. _file:


File Objects
============

File objects serve as your entry point into the world of HDF5.  In addition
to the File-specific capabilities listed here, every File instance is
also an :ref:`HDF5 group <group>` representing the `root group` of the file.

.. _file_open:

Opening & creating files
------------------------

HDF5 files work generally like standard Python file objects.  They support
standard modes like r/w/a, and should be closed when they are no longer in
use.  However, there is obviously no concept of "text" vs "binary" mode.

    >>> f = h5py.File('myfile.hdf5','r')

The file name may be a byte string or unicode string. Valid modes are:

    ========  ================================================
     r        Readonly, file must exist (default)
     r+       Read/write, file must exist
     w        Create file, truncate if exists
     w- or x  Create file, fail if exists
     a        Read/write if exists, create otherwise
    ========  ================================================

.. versionchanged:: 3.0
   Files are now opened read-only by default. Earlier versions of h5py would
   pick different modes depending on the presence and permissions of the file.

.. _file_driver:

File drivers
------------

HDF5 ships with a variety of different low-level drivers, which map the logical
HDF5 address space to different storage mechanisms.  You can specify which
driver you want to use when the file is opened::

    >>> f = h5py.File('myfile.hdf5', driver=<driver name>, <driver_kwds>)

For example, the HDF5 "core" driver can be used to create a purely in-memory
HDF5 file, optionally written out to disk when it is closed.  Here's a list
of supported drivers and their options:

    None
        **Strongly recommended.** Use the standard HDF5 driver appropriate
        for the current platform. On UNIX, this is the H5FD_SEC2 driver;
        on Windows, it is H5FD_WINDOWS.

    'sec2'
        Unbuffered, optimized I/O using standard POSIX functions.

    'stdio'
        Buffered I/O using functions from stdio.h.

    'core'
        Store and manipulate the data in memory, and optionally write it
        back out when the file is closed. Using this with an existing file
        and a reading mode will read the entire file into memory. Keywords:

        backing_store:
          If True (default), save changes to the real file at the specified
          path on :meth:`~.File.close` or :meth:`~.File.flush`.
          If False, any changes are discarded when the file is closed.

        block_size:
          Increment (in bytes) by which memory is extended. Default is 64k.

    'family'
        Store the file on disk as a series of fixed-length chunks.  Useful
        if the file system doesn't allow large files.  Note: the filename
        you provide *must* contain a printf-style integer format code
        (e.g. %d"), which will be replaced by the file sequence number.
        Keywords:

        memb_size:  Maximum file size (default is 2**31-1).

    'fileobj'
        Store the data in a Python file-like object; see below.
        This is the default if a file-like object is passed to :class:`File`.

    'split'
        Splits the meta data and raw data into separate files. Keywords:

        meta_ext:
          Metadata filename extension. Default is '-m.h5'.

        raw_ext:
          Raw data filename extension. Default is '-r.h5'.

    'ros3'
        Allows read only access to HDF5 files on S3. Keywords:

        aws_region:
          Name of the AWS "region" where the S3 bucket with the file is, e.g. ``b"us-east-1"``. Default is ``b''``.

        secret_id:
          "Access ID" for the resource. Default is ``b''``.

        secret_key:
          "Secret Access Key" associated with the ID and resource. Default is ``b''``.

        The argument values must be ``bytes`` objects.


.. _file_fileobj:

Python file-like objects
------------------------

.. versionadded:: 2.9

The first argument to :class:`File` may be a Python file-like object, such as
an :class:`io.BytesIO` or :class:`tempfile.TemporaryFile` instance.
This is a convenient way to create temporary HDF5 files, e.g. for testing or to
send over the network.

The file-like object must be open for binary I/O, and must have these methods:
``read()`` (or ``readinto()``), ``write()``, ``seek()``, ``tell()``,
``truncate()`` and ``flush()``.


    >>> tf = tempfile.TemporaryFile()
    >>> f = h5py.File(tf, 'w')

Accessing the :class:`File` instance after the underlying file object has been
closed will result in undefined behaviour.

When using an in-memory object such as :class:`io.BytesIO`, the data written
will take up space in memory. If you want to write large amounts of data,
a better option may be to store temporary data on disk using the functions in
:mod:`tempfile`.

.. literalinclude:: ../../examples/bytesio.py

.. warning::

   When using a Python file-like object for an HDF5 file, make sure to close
   the HDF5 file before closing the file object it's wrapping. If there is an
   error while trying to close the HDF5 file, segfaults may occur.

.. note::

   Using a Python file-like object for HDF5 is internally more complex,
   as the HDF5 C code calls back into Python to access it. It inevitably
   has more ways to go wrong, and the failures may be less clear when it does.
   For some common use cases, you can easily avoid it:

   - To create a file in memory and never write it to disk, use the ``'core'``
     driver with ``mode='w', backing_store=False`` (see :ref:`file_driver`).
   - To use a temporary file securely, make a temporary directory and
     :ref:`open a file path <file_open>` inside it.

.. _file_version:

Version bounding
----------------

HDF5 has been evolving for many years now.  By default, the library will write
objects in the most compatible fashion possible, so that older versions will
still be able to read files generated by modern programs.  However, there can be
feature or performance advantages if you are willing to forgo a certain level of
backwards compatibility.  By using the "libver" option to :class:`File`, you can
specify the minimum and maximum sophistication of these structures:

    >>> f = h5py.File('name.hdf5', libver='earliest') # most compatible
    >>> f = h5py.File('name.hdf5', libver='latest')   # most modern

Here "latest" means that HDF5 will always use the newest version of these
structures without particular concern for backwards compatibility.  The
"earliest" option means that HDF5 will make a *best effort* to be backwards
compatible.

The default is "earliest".

Specifying version bounds has changed from HDF5 version 1.10.2. There are two new
compatibility levels: `v108` (for HDF5 1.8) and `v110` (for HDF5 1.10). This
change enables, for example, something like this:

    >>> f = h5py.File('name.hdf5', libver=('earliest', 'v108'))

which enforces full backward compatibility up to HDF5 1.8. Using any HDF5
feature that requires a newer format will raise an error.

`latest` is now an alias to another bound label that represents the latest
version. Because of this, the `File.libver` property will not use `latest` in
its output for HDF5 1.10.2 or later.

.. _file_closing:

Closing files
-------------

If you call :meth:`File.close`, or leave a ``with h5py.File(...)`` block,
the file will be closed and any objects (such as groups or datasets) you have
from that file will become unusable. This is equivalent to what HDF5 calls
'strong' closing.

If a file object goes out of scope in your Python code, the file will only
be closed when there are no remaining objects belonging to it. This is what
HDF5 calls 'weak' closing.

.. code-block::

    with h5py.File('f1.h5', 'r') as f1:
        ds = f1['dataset']

    # ERROR - can't access dataset, because f1 is closed:
    ds[0]

    def get_dataset():
        f2 = h5py.File('f2.h5', 'r')
        return f2['dataset']
    ds = get_dataset()

    # OK - f2 is out of scope, but the dataset reference keeps it open:
    ds[0]

    del ds  # Now f2.h5 will be closed


.. _file_userblock:

User block
----------

HDF5 allows the user to insert arbitrary data at the beginning of the file,
in a reserved space called the `user block`.  The length of the user block
must be specified when the file is created.  It can be either zero
(the default) or a power of two greater than or equal to 512.  You
can specify the size of the user block when creating a new file, via the
``userblock_size`` keyword to File; the userblock size of an open file can
likewise be queried through the ``File.userblock_size`` property.

Modifying the user block on an open file is not supported; this is a limitation
of the HDF5 library.  However, once the file is closed you are free to read and
write data at the start of the file, provided your modifications don't leave
the user block region.


.. _file_filenames:

Filenames on different systems
------------------------------

Different operating systems (and different file systems) store filenames with
different encodings. Additionally, in Python there are at least two different
representations of filenames, as encoded bytes (via str on Python 2, bytes on
Python 3) or as a unicode string (via unicode on Python 2 and str on Python 3).
The safest bet when creating a new file is to use unicode strings on all
systems.

macOS (OSX)
...........
macOS is the simplest system to deal with, it only accepts UTF-8, so using
unicode paths will just work (and should be preferred).

Linux (and non-macOS Unix)
..........................
Unix-like systems use locale settings to determine the correct encoding to use.
These are set via a number of different environment variables, of which ``LANG``
and ``LC_ALL`` are the ones of most interest. Of special interest is the ``C``
locale, which Python will interpret as only allowing ASCII, meaning unicode
paths should be pre-encoded. This will likely change in Python 3.7 with
https://www.python.org/dev/peps/pep-0538/, but this will likely be backported by
distributions to earlier versions.

To summarise, use unicode strings where possible, but be aware that sometimes
using encoded bytes may be necessary to read incorrectly encoded filenames.

Windows
.......
Windows systems have two different APIs to perform file-related operations, a
ANSI (char, legacy) interface and a unicode (wchar) interface. HDF5 currently
only supports the ANSI interface, which is limited in what it can encode. This
means that it may not be possible to open certain files, and because
:ref:`group_extlinks` do not specify their encoding, it is possible that opening an
external link may not work. There is work being done to fix this (see
https://github.com/h5py/h5py/issues/839), but it is likely there will need to be
breaking changes make to allow Windows to have the same level of support for
unicode filenames as other operating systems.

The best suggestion is to use unicode strings, but to keep to ASCII for
filenames to avoid possible breakage.


.. _file_cache:

Chunk cache
-----------

:ref:`dataset_chunks` allows datasets to be stored on disk in separate pieces.
When a part of any one of these pieces is needed, the entire chunk is read into
memory before the requested part is copied to the user's buffer.  To the extent
possible those chunks are cached in memory, so that if the user requests a
different part of a chunk that has already been read, the data can be copied
directly from memory rather than reading the file again.  The details of a
given dataset's chunks are controlled when creating the dataset, but it is
possible to adjust the behavior of the chunk *cache* when opening the file.

The parameters controlling this behavior are prefixed by ``rdcc``, for *raw data
chunk cache*.

* ``rdcc_nbytes`` sets the total size (measured in bytes) of the raw data chunk
  cache for each dataset.  The default size is 1 MB.
  This should be set to the size of each chunk times the number of
  chunks that are likely to be needed in cache.
* ``rdcc_w0`` sets the policy for chunks to be
  removed from the cache when more space is needed.  If the value is set to 0,
  then the library will always evict the least recently used chunk in cache.  If
  the value is set to 1, the library will always evict the least recently used
  chunk which has been fully read or written, and if none have been fully read
  or written, it will evict the least recently used chunk.  If the value is
  between 0 and 1, the behavior will be a blend of the two.  Therefore, if the
  application will access the same data more than once, the value should be set
  closer to 0, and if the application does not, the value should be set closer
  to 1.
* ``rdcc_nslots`` is the number of chunk slots in
  the cache for this entire file.  In order to allow the chunks to be looked up
  quickly in cache, each chunk is assigned a unique hash value that is used to
  look up the chunk.  The cache contains a simple array of pointers to chunks,
  which is called a hash table.  A chunk's hash value is simply the index into
  the hash table of the pointer to that chunk.  While the pointer at this
  location might instead point to a different chunk or to nothing at all, no
  other locations in the hash table can contain a pointer to the chunk in
  question.  Therefore, the library only has to check this one location in the
  hash table to tell if a chunk is in cache or not.  This also means that if two
  or more chunks share the same hash value, then only one of those chunks can be
  in the cache at the same time.  When a chunk is brought into cache and another
  chunk with the same hash value is already in cache, the second chunk must be
  evicted first.  Therefore it is very important to make sure that the size of
  the hash table (which is determined by the ``rdcc_nslots`` parameter) is large
  enough to minimize the number of hash value collisions.  Due to the hashing
  strategy, this value should ideally be a prime number.  As a rule of thumb,
  this value should be at least 10 times the number of chunks that can fit in
  ``rdcc_nbytes`` bytes. For maximum performance, this value should be set
  approximately 100 times that number of chunks. The default value is 521.

Chunks and caching are described in greater detail in the `HDF5 documentation
<https://portal.hdfgroup.org/display/HDF5/Chunking+in+HDF5>`_.


Reference
---------

.. note::

    Unlike Python file objects, the attribute :attr:`File.name` gives the
    HDF5 name of the root group, "``/``". To access the on-disk name, use
    :attr:`File.filename`.

.. class:: File(name, mode=None, driver=None, libver=None, \
    userblock_size=None, swmr=False, rdcc_nslots=None, rdcc_nbytes=None, \
    rdcc_w0=None, track_order=None, fs_strategy=None, fs_persist=False, \
    fs_threshold=1, **kwds)

    Open or create a new file.

    Note that in addition to the File-specific methods and properties listed
    below, File objects inherit the full interface of :class:`Group`.

    :param name:    Name of file (`bytes` or `str`), or an instance of
                    :class:`h5f.FileID` to bind to an existing
                    file identifier, or a file-like object
                    (see :ref:`file_fileobj`).
    :param mode:    Mode in which to open file; one of
                    ("w", "r", "r+", "a", "w-").  See :ref:`file_open`.
    :param driver:  File driver to use; see :ref:`file_driver`.
    :param libver:  Compatibility bounds; see :ref:`file_version`.
    :param userblock_size:  Size (in bytes) of the user block.  If nonzero,
                    must be a power of 2 and at least 512.  See
                    :ref:`file_userblock`.
    :param swmr:    If ``True`` open the file in single-writer-multiple-reader
                    mode. Only used when mode="r".
    :param rdcc_nbytes:  Total size of the raw data chunk cache in bytes. The
                    default size is :math:`1024^2` (1 MiB) per dataset.
    :param rdcc_w0: Chunk preemption policy for all datasets.  Default value is
                    0.75.
    :param rdcc_nslots:  Number of chunk slots in the raw data chunk cache for
                    this file.  Default value is 521.
    :param track_order:  Track dataset/group/attribute creation order under
                    root group if ``True``.  Default is
                    ``h5.get_config().track_order``.
    :param fs_strategy: The file space handling strategy to be used.
            Only allowed when creating a new file. One of "fsm", "page",
            "aggregate", "none", or None (to use the HDF5 default).
    :param fs_persist: A boolean to indicate whether free space should be
            persistent or not. Only allowed when creating a new file. The
            default is False.
    :param fs_threshold: The smallest free-space section size that the free
            space manager will track. Only allowed when creating a new file.
            The default is 1.
    :param kwds:    Driver-specific keywords; see :ref:`file_driver`.

    .. method:: __bool__()

        Check that the file descriptor is valid and the file open:

            >>> f = h5py.File(filename)
            >>> f.close()
            >>> if f:
            ...     print("file is open")
            ... else:
            ...     print("file is closed")
            file is closed

    .. method:: close()

        Close this file.  All open objects will become invalid.

    .. method:: flush()

        Request that the HDF5 library flush its buffers to disk.

    .. attribute:: id

        Low-level identifier (an instance of :class:`FileID <low:h5py.h5f.FileID>`).

    .. attribute:: filename

        Name of this file on disk.  Generally a Unicode string; a byte string
        will be used if HDF5 returns a non-UTF-8 encoded string.

    .. attribute:: mode

        String indicating if the file is open readonly ("r") or read-write
        ("r+").  Will always be one of these two values, regardless of the
        mode used to open the file.

    .. attribute:: swmr_mode

       True if the file access is using :doc:`/swmr`. Use :attr:`mode` to
       distinguish SWMR read from write.

    .. attribute:: driver

        String giving the driver used to open the file.  Refer to
        :ref:`file_driver` for a list of drivers.

    .. attribute:: libver

        2-tuple with library version settings.  See :ref:`file_version`.

    .. attribute:: userblock_size

        Size of user block (in bytes).  Generally 0.  See :ref:`file_userblock`.
