Module H5FD
===========

.. automodule:: h5py.h5fd

Module constants
----------------

Memory usage types for MULTI file driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: MEM_DEFAULT
.. data:: MEM_SUPER
.. data:: MEM_BTREE
.. data:: MEM_DRAW
.. data:: MEM_GHEAP
.. data:: MEM_LHEAP
.. data:: MEM_OHDR
.. data:: MEM_NTYPES


Data transfer modes for MPIO driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: MPIO_INDEPENDENT
.. data:: MPIO_COLLECTIVE

File drivers types
~~~~~~~~~~~~~~~~~~

.. data:: CORE
.. data:: FAMILY
.. data:: LOG
.. data:: MPIO
.. data:: MULTI
.. data:: SEC2
.. data:: STDIO
.. data:: WINDOWS

Logging driver settings
~~~~~~~~~~~~~~~~~~~~~~~

.. note:: Not all logging flags are currently implemented by HDF5.

.. data:: LOG_LOC_READ
.. data:: LOG_LOC_WRITE
.. data:: LOG_LOC_SEEK
.. data:: LOG_LOC_IO

.. data:: LOG_FILE_READ
.. data:: LOG_FILE_WRITE
.. data:: LOG_FILE_IO

.. data:: LOG_FLAVOR

.. data:: LOG_NUM_READ
.. data:: LOG_NUM_WRITE
.. data:: LOG_NUM_SEEK
.. data:: LOG_NUM_IO

.. data:: LOG_TIME_OPEN
.. data:: LOG_TIME_READ
.. data:: LOG_TIME_WRITE
.. data:: LOG_TIME_SEEK
.. data:: LOG_TIME_CLOSE
.. data:: LOG_TIME_IO

.. data:: LOG_ALLOC
.. data:: LOG_ALL
