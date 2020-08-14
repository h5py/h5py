Module H5D
==========

.. automodule:: h5py.h5d

Functional API
--------------

.. autofunction:: open
.. autofunction:: create

Dataset Objects
---------------

.. autoclass:: DatasetID
   :members:

Module constants
----------------

Storage strategies
~~~~~~~~~~~~~~~~~~

.. data:: COMPACT
.. data:: CONTIGUOUS
.. data:: CHUNKED

.. _ref.h5d.ALLOC_TIME:

Allocation times
~~~~~~~~~~~~~~~~

.. data:: ALLOC_TIME_DEFAULT
.. data:: ALLOC_TIME_LATE
.. data:: ALLOC_TIME_EARLY
.. data:: ALLOC_TIME_INCR

Allocation status
~~~~~~~~~~~~~~~~~

.. data:: SPACE_STATUS_NOT_ALLOCATED
.. data:: SPACE_STATUS_PART_ALLOCATED
.. data:: SPACE_STATUS_ALLOCATED

Fill time
~~~~~~~~~

.. data:: FILL_TIME_ALLOC
.. data:: FILL_TIME_NEVER
.. data:: FILL_TIME_IFSET

Fill values
~~~~~~~~~~~

.. data:: FILL_VALUE_UNDEFINED
.. data:: FILL_VALUE_DEFAULT
.. data:: FILL_VALUE_USER_DEFINED
