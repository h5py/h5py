Module H5O
==========

.. automodule:: h5py.h5o

Functional API
--------------

.. autofunction:: open
.. autofunction:: link
.. autofunction:: copy
.. autofunction:: set_comment
.. autofunction:: get_comment
.. autofunction:: visit
.. autofunction:: get_info

Info classes
------------

.. autoclass:: ObjInfo
    :members:

Module constants
----------------

Object types
~~~~~~~~~~~~

.. data:: TYPE_GROUP
.. data:: TYPE_DATASET
.. data:: TYPE_NAMED_DATATYPE

.. _ref.h5o.COPY:

Copy flags
~~~~~~~~~~

.. data:: COPY_SHALLOW_HIERARCHY_FLAG

    Copy only immediate members of a group.

.. data:: COPY_EXPAND_SOFT_LINK_FLAG

    Expand soft links into new objects.

.. data:: COPY_EXPAND_EXT_LINK_FLAG

    Expand external link into new objects.

.. data:: COPY_EXPAND_REFERENCE_FLAG

    Copy objects that are pointed to by references.

.. data:: COPY_WITHOUT_ATTR_FLAG

    Copy object without copying attributes.

