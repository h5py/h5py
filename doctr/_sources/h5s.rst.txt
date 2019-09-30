Module H5S
==========

.. automodule:: h5py.h5s

Functional API
--------------

.. autofunction:: create
.. autofunction:: create_simple
.. autofunction:: decode

Dataspace objects
-----------------

.. autoclass:: SpaceID
    :show-inheritance:
    :members:

Module constants
----------------

.. data:: ALL

    Accepted in place of an actual dataspace; means "every point"

.. data:: UNLIMITED

    Indicates an unlimited maximum dimension

Dataspace class codes
~~~~~~~~~~~~~~~~~~~~~

.. data:: NO_CLASS
.. data:: SCALAR
.. data:: SIMPLE

Selection codes
~~~~~~~~~~~~~~~

.. data:: SELECT_NOOP
.. data:: SELECT_SET
.. data:: SELECT_OR
.. data:: SELECT_AND
.. data:: SELECT_XOR
.. data:: SELECT_NOTB
.. data:: SELECT_NOTA
.. data:: SELECT_APPEND
.. data:: SELECT_PREPEND
.. data:: SELECT_INVALID

Existing selection type
~~~~~~~~~~~~~~~~~~~~~~~

.. data:: SEL_NONE
.. data:: SEL_POINTS
.. data:: SEL_HYPERSLABS
.. data:: SEL_ALL
