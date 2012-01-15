from defs cimport *

cdef class ObjectID:

    cdef object __weakref__
    cdef readonly hid_t id
    cdef public int locked
    cdef object _hash

# Convenience functions
cdef hid_t pdefault(ObjectID pid)

# Inheritance scheme (for top-level cimport and import statements):
#
# _objects, _proxy, h5fd, h5z
# h5i, h5r, utils
# _conv, h5t, h5s
# h5p
# h5d, h5a, h5f, h5g
# h5l

