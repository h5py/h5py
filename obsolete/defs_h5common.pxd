from defs_h5 cimport hsize_t

cdef hsize_t* tuple_to_dims(object dims_tpl, int rank)
cdef object dims_to_tuple(hsize_t* dims, int rank)

