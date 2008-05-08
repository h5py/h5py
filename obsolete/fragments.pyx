def set_fill_value(hid_t plist, object value):
    """ (INT plist, INT type_id, ARRAY_SCALAR value)
        For lists of class CLASS_DATASET_CREATE

        Set the fill value for the dataset. <value> should be a NumPy array 
        scalar or 0-dimensional array.  It's up to you to make sure the dtype 
        of the scalar is compatible with the type of whatever dataset you want 
        to use this list on.

        As a special exception, providing a value of None means the fill is
        undefined (HDF5 default is otherwise zero-fill).
    """
    cdef hid_t type_id
    cdef herr_t retval
    cdef void* data_ptr

    raise NotImplementedError()

    if value is None:
        retval = H5Pset_fill_value(plist, 0, NULL)
        if retval < 0:
            raise PropertyError("Failed to undefine fill value on list %d" % plist)
        return

    if not PyArray_CheckScalar(value):
        raise ValueError("Given fill value must be a Numpy array scalar or 0-dimensional array")

    data_ptr = malloc(128)
    PyArray_ScalarAsCtype(value, data_ptr)
    type_id = h5t.py_dtype_to_h5t(value.dtype)

    retval = H5Pset_fill_value(plist, type_id, data_ptr)
    if retval < 0:
        free(data_ptr)
        H5Tclose(type_id)
        raise PropertyError("Failed to set fill value on list %d to %s" % (plist, repr(value)))

    free(data_ptr)
    H5Tclose(type_id)

def get_fill_value(hid_t plist, object dtype_in):
    """ (INT plist_id, DTYPE dtype_in) => ARRAY_SCALAR value

        Obtain the fill value.  Due to restrictions in the HDF5 library
        design, you have to provide a Numpy dtype object specifying the
        fill value type.  The function will raise an exception if this
        type is not conversion-compatible with the fill value type recorded
        in the list.
    """
    raise NotImplementedError()
    cdef herr_t retval
    cdef hid_t type_id

def set_filter(hid_t plist, int filter_code, unsigned int flags, object data_tpl=()):
    """ (INT plist_id, INT filter_type_code, UINT flags, TUPLE data)
    """

    cdef unsigned int *data
    cdef size_t datalen
    cdef int i
    cdef herr_t retval

    if !PyTuple_Check(data_tpl):
        raise ValueError("Data for the filter must be a tuple of integers")

    datalen = len(data_tpl)
    data = <unsigned int*>malloc(sizeof(unsigned int)*datalen)

    try:
        for i from 0<=i<datalen:
            data[i] = data_tpl[i]

        retval = H5Pset_filter(plist, filter_code, flags, data_len, data)
        if retval < 0:
            raise PropertyError("Failed to set filter code %d on list %d; flags %d, data %s" % (filter_code, plist, flags, str(data_tpl)))
    finally:
        free(data) 
    
def all_filters_avail(hid_t plist):

    cdef htri_t retval
    retval = H5Pall_filters_avail(plist)
    if retval < 0:
        raise PropertyError("Failed to determine filter status on list %d" % plist)
    return bool(retval)

def get_nfilters(hid_t plist)

    cdef int retval
    retval = H5Pget_nfilters(plist)
    if retval < 0:
        raise PropertyError("Failed to determine number of filters in list %d" % plist)
    return retval

cdef class FilterInfo:

    cdef object name
    cdef int code
    cdef unsigned int flags
    cdef object data

def get_filter_info(hid_t plist, unsigned int filter_no):

    cdef char namearr[256]
    cdef int namelen
    cdef unsigned int flags
    cdef size_t datalen
    cdef unsigned int data[256]
    cdef int retval
    cdef int i

    datalen = 256
    namelen = 256

    retval = <int>H5Pget_filter(plist, filter_no, &flags, &datalen, &data, namelen, &namearr)
    if retval < 0:
        raise PropertyError("Failed to get info for filter %d on list %d" % (filter_no, plist))
    
    # HDF5 docs claim the string may not be properly terminated.
    for i from 0<=i<namelen:
        if namearr[i] == c'\0':
            break
    if i == namelen:
        namearr[namelen-1] = c'\0'

    tpl = PyTuple_New(datalen)
    for i from 0<=i<datalen:
        tmp = data[i]
        Py_INCREF(tmp)  # to get around pyrex issues
        PyTuple_SetItem(tpl, i, tmp)

    info = FilterInfo()
    info.name = &namearr
    info.code = retval
    info.flags = flags
    info.data = tpl

    return info
