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

Other constants
~~~~~~~~~~~~~~~

.. data:: CLOSE_WEAK
.. data:: CLOSE_SEMI
.. data:: CLOSE_STRONG
.. data:: CLOSE_DEFAULT

.. data:: SCOPE_LOCAL
.. data:: SCOPE_GLOBAL

.. data:: OBJ_FILE
.. data:: OBJ_DATASET
.. data:: OBJ_GROUP
.. data:: OBJ_DATATYPE
.. data:: OBJ_ATTR
.. data:: OBJ_ALL
.. data:: OBJ_LOCAL


