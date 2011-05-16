from defs cimport *

cdef class H5PYConfig:

    cdef readonly object _r_name
    cdef readonly object _i_name
    cdef readonly object _f_name
    cdef readonly object _t_name
    cdef readonly object API_16
    cdef readonly object API_18

cpdef H5PYConfig get_config()


