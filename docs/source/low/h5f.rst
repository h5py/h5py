Module H5F
==========

.. automodule:: h5py.h5f

Functional API
--------------

.. autofunction:: open
.. autofunction:: create
.. autofunction:: flush
.. autofunction:: is_hdf5
.. autofunction:: mount
.. autofunction:: unmount
.. autofunction:: get_name
.. autofunction:: get_obj_count
.. autofunction:: get_obj_ids

File objects
------------

.. autoclass:: FileID
    :members:

Module constants
----------------

.. _ref.h5f.ACC:

File access flags
~~~~~~~~~~~~~~~~~

.. data:: ACC_TRUNC

    Create/truncate file

.. data:: ACC_EXCL

    Create file if it doesn't exist; fail otherwise

.. data:: ACC_RDWR

    Open in read/write mode

.. data:: ACC_RDONLY

    Open in read-only mode


.. _ref.h5f.CLOSE:

File close strength
~~~~~~~~~~~~~~~~~~~

.. data:: CLOSE_WEAK
.. data:: CLOSE_SEMI
.. data:: CLOSE_STRONG
.. data:: CLOSE_DEFAULT

.. _ref.h5f.SCOPE:

File scope
~~~~~~~~~~

.. data:: SCOPE_LOCAL
.. data:: SCOPE_GLOBAL

.. _ref.h5f.OBJ:

Object types
~~~~~~~~~~~~

.. data:: OBJ_FILE
.. data:: OBJ_DATASET
.. data:: OBJ_GROUP
.. data:: OBJ_DATATYPE
.. data:: OBJ_ATTR
.. data:: OBJ_ALL
.. data:: OBJ_LOCAL

Library version bounding
~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: LIBVER_EARLIEST
.. data:: LIBVER_LATEST


