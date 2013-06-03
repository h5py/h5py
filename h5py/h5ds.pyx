#+
#
# This file is part of h5py, a low-level Python interface to the HDF5 library.
#
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
#
# $Date$
#
#-

"""
    Low-level HDF5 "H5G" group interface.
"""

# Compile-time imports
from h5d cimport DatasetID
from utils cimport emalloc, efree


def set_scale(DatasetID dset not None, char* dimname=''):
    """(DatasetID dset, STRING dimname)

    Convert dataset dset to a dimension scale, with optional name dimname.
    """
    H5DSset_scale(dset.id, dimname)

def is_scale(DatasetID dset not None):
    """(DatasetID dset)

    Determines whether dset is a dimension scale.
    """
    return <bint>(H5DSis_scale(dset.id))

def attach_scale(DatasetID dset not None, DatasetID dscale not None, unsigned
                 int idx):
    """(DatasetID dset, DatasetID dscale, UINT idx)

    Attach a scale dataset dscale to main dataset dset, on axis idx.
    """
    H5DSattach_scale(dset.id, dscale.id, idx)

def is_attached(DatasetID dset not None, DatasetID dscale not None,
                unsigned int idx):
    """(DatasetID dset, DatasetID dscale, UINT idx) => BOOL

    Determine whether dimension scale dataset dscale is attached to axis
    idx of main dataset dset.
    """
    return <bint>(H5DSis_attached(dset.id, dscale.id, idx))

def detach_scale(DatasetID dset not None, DatasetID dscale not None,
                 unsigned int idx):
    """(DatasetID dset, DatasetID dscale, UINT idx)

    Remove dimension scale dataset dscale from axis idx of main dataset dset.
    """
    H5DSdetach_scale(dset.id, dscale.id, idx)

def get_num_scales(DatasetID dset not None, unsigned int dim):
    """(DatasetID dset, UINT dim) => INT

    Determine number of scales attached to axis dim of dataset dset.
    """ 
    return H5DSget_num_scales(dset.id, dim)

def set_label(DatasetID dset not None, unsigned int idx, char* label):
    """(DatasetID dset, UINT idx, STRING label)

    Label axis idx of dataset dset.
    """
    H5DSset_label(dset.id, idx, label)

def get_label(DatasetID dset not None, unsigned int idx):
    """(DatasetID dset, UINT idx) => STRING name

    Get the label attached to axis idx of dataset dset.

    Returns "" if no label has been set.
    """
    cdef ssize_t size
    cdef char* label
    label = NULL

    size = H5DSget_label(dset.id, idx, NULL, 0)
    if size <= 0:
        return ''
    label = <char*>emalloc(sizeof(char)*(size+1))
    try:
        H5DSget_label(dset.id, idx, label, size+1)
        plabel = label
        return plabel
    finally:
        efree(label)

def get_scale_name(DatasetID dscale not None):
    """(DatasetID dscale) => STRING name

    Get the name associated with dimension scale dataset dset.
  
    Returns "" if no name has been set.
    """
    cdef ssize_t namelen
    cdef char* name = NULL

    namelen = H5DSget_scale_name(dscale.id, NULL, 0)
    if namelen <= 0:
        return ''
    name = <char*>emalloc(sizeof(char)*(namelen+1))
    try:
        H5DSget_scale_name(dscale.id, name, namelen+1)
        pname = name
        return pname
    finally:
        efree(name)


cdef class _DimensionScaleVisitor:

    cdef object func
    cdef object retval

    def __init__(self, func):
        self.func = func
        self.retval = None


cdef herr_t cb_ds_iter(hid_t dset, unsigned int dim, hid_t scale, void* vis_in) except 2:

    cdef _DimensionScaleVisitor vis = <_DimensionScaleVisitor>vis_in

    # we did not retrieve the scale identifier using the normal machinery,
    # so we need to inc_ref it before using it to create a DatasetID.
    H5Iinc_ref(scale)
    vis.retval = vis.func(DatasetID.open(scale))

    if vis.retval is not None:
        return 1
    return 0


def iterate(DatasetID dset not None, unsigned int dim, object func,
            int startidx=0):
    """ (DatasetID loc, UINT dim, CALLABLE func, UINT startidx=0)
    => Return value from func

    Iterate a callable (function, method or callable object) over the
    dimension scales attached to axis dim of dataset dset.  
    Your callable shoutld have the signature::

        func(DatasetID dscale) => Result

    Returning None continues iteration; returning anything else aborts
    iteration and returns that value.
    """
    if startidx < 0:
        raise ValueError("Starting index must be non-negative")

    cdef int i = startidx
    cdef _DimensionScaleVisitor vis = _DimensionScaleVisitor(func)

    H5DSiterate_scales(dset.id, dim, &i, <H5DS_iterate_t>cb_ds_iter, <void*>vis)

    return vis.retval
