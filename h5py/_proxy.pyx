# cython: profile=False

# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Proxy functions for read/write, to work around the HDF5 bogus type issue.
"""

include "config.pxi"

cdef enum copy_dir:
    H5PY_SCATTER = 0,
    H5PY_GATHER

cdef herr_t attr_rw(hid_t attr, hid_t mtype, void *progbuf, int read) except -1:

    cdef htri_t need_bkg
    cdef hid_t atype = -1
    cdef hid_t aspace = -1
    cdef hsize_t npoints

    cdef size_t msize, asize
    cdef void* conv_buf = NULL
    cdef void* back_buf = NULL

    try:
        atype = H5Aget_type(attr)

        if not (needs_proxy(atype) or needs_proxy(mtype)):
            if read:
                H5Aread(attr, mtype, progbuf)
            else:
                H5Awrite(attr, mtype, progbuf)

        else:

            asize = H5Tget_size(atype)
            msize = H5Tget_size(mtype)
            aspace = H5Aget_space(attr)
            npoints = H5Sget_select_npoints(aspace)

            conv_buf = create_buffer(asize, msize, npoints)

            if read:
                need_bkg = needs_bkg_buffer(atype, mtype)
            else:
                need_bkg = needs_bkg_buffer(mtype, atype)
            if need_bkg:
                back_buf = create_buffer(msize, asize, npoints)
                if read:
                    memcpy(back_buf, progbuf, msize*npoints)

            if read:
                H5Aread(attr, atype, conv_buf)
                H5Tconvert(atype, mtype, npoints, conv_buf, back_buf, H5P_DEFAULT)
                memcpy(progbuf, conv_buf, msize*npoints)
            else:
                memcpy(conv_buf, progbuf, msize*npoints)
                H5Tconvert(mtype, atype, npoints, conv_buf, back_buf, H5P_DEFAULT)
                H5Awrite(attr, atype, conv_buf)
                H5Dvlen_reclaim(atype, aspace, H5P_DEFAULT, conv_buf)

    finally:
        free(conv_buf)
        free(back_buf)
        if atype > 0:
            H5Tclose(atype)
        if aspace > 0:
            H5Sclose(aspace)

    return 0


# =============================================================================
# Proxy for vlen buf workaround

cdef herr_t dset_rw(size_t count, hid_t* dset, hid_t* mtype, hid_t* mspace, hid_t* _fspace,
    hid_t dxpl, void **progbuf, int read) except -1:

    cdef hid_t plist_id = -1
    cdef hid_t rawdstype = -1

    cdef hid_t* dstype = NULL # Dataset datatype
    cdef hid_t* cspace = NULL # Temporary contiguous dataspaces
    cdef hid_t* mspace_tmp = NULL
    cdef hid_t* fspace_tmp = NULL

    cdef htri_t* need_bkg = NULL

    cdef void** back_buf = NULL
    cdef void** conv_buf = NULL

    cdef hsize_t* npoints = NULL

    cdef bint rw_needs_proxy = False

    try:
        # Make local list of mem/file spaces which may be freely modified
        mspace_tmp = <hid_t*>malloc(sizeof(hid_t*) * count)
        fspace_tmp = <hid_t*>malloc(sizeof(hid_t*) * count)
        dstype = <hid_t*> malloc(sizeof(hid_t*) * count)

        for i in range(count):
            mspace_tmp[i] = mspace[i]
            fspace_tmp[i] = _fspace[i]

        # Issue 372: when a compound type is involved, using the dataset type
        # may result in uninitialized data being sent to H5Tconvert for fields
        # not present in the memory type.  Limit the type used for the dataset
        # to only those fields present in the memory type.  We can't use the
        # memory type directly because of course that triggers HDFFV-1063.
        for i in range(count):
            if (H5Tget_class(mtype[i]) == H5T_COMPOUND) and (not read):
                rawdstype = H5Dget_type(dset[i])
                dstype[i] = make_reduced_type(mtype[i], rawdstype)
                H5Tclose(rawdstype)
                rawdstype = -1
            else:
                dstype[i] = H5Dget_type(dset[i])

            rw_needs_proxy = rw_needs_proxy or (needs_proxy(dstype[i]) or needs_proxy(mtype[i]))

        if not rw_needs_proxy:
            if read:
                if count > 1:
                    H5Dread_multi(count, <hid_t*> dset, <hid_t*>mtype, <hid_t*> mspace_tmp, <hid_t*>fspace_tmp, dxpl, progbuf)
                else:
                    H5Dread(dset[0], mtype[0], mspace_tmp[0], fspace_tmp[0], dxpl, <void*>progbuf[0])
            else:
                if count > 1:
                    H5Dwrite_multi(count, <hid_t*> dset, <hid_t*>mtype, <hid_t*> mspace_tmp, <hid_t*>fspace_tmp, dxpl, <const void**> progbuf)
                else:
                    H5Dwrite(dset[0], mtype[0],mspace_tmp[0], fspace_tmp[0], dxpl, <void*>progbuf[0])
        else:
            cspace = <hid_t*> malloc(sizeof(hid_t*) * count)
            need_bkg = <htri_t*> malloc(sizeof(htri_t) * count)
            back_buf = <void**> malloc(sizeof(void*) * count)
            conv_buf = <void**> malloc(sizeof(void*) * count)
            npoints = <hsize_t*> malloc(sizeof(hsize_t) * count)

            for i in range(count):
                back_buf[i] = NULL
                conv_buf[i] = NULL

            for i in range(count):
                if mspace_tmp[i] == H5S_ALL and fspace_tmp[i] != H5S_ALL:
                   mspace_tmp[i] = fspace_tmp[i]
                elif mspace_tmp[i] != H5S_ALL and fspace_tmp[i] == H5S_ALL:
                    fspace_tmp[i] = mspace_tmp[i]
                elif mspace_tmp[i] == H5S_ALL and fspace_tmp[i] == H5S_ALL:
                   mspace_tmp[i] = fspace_tmp[i] = H5Dget_space(dset[i])

                npoints[i] = H5Sget_select_npoints(mspace_tmp[i])
                cspace[i] = H5Screate_simple(1, <hsize_t*> &npoints[i], NULL)

                conv_buf[i] = create_buffer(H5Tget_size(dstype[i]), H5Tget_size(mtype[i]), npoints[i])

                # Only create a (contiguous) backing buffer if absolutely
                # necessary. Note this buffer always has memory type.
                if read:
                    need_bkg[i] = needs_bkg_buffer(dstype[i], mtype[i])
                else:
                    need_bkg[i] = needs_bkg_buffer(mtype[i], dstype[i])

                if need_bkg[i]:
                    back_buf[i] = create_buffer(H5Tget_size(dstype[i]), H5Tget_size(mtype[i]), npoints[i])
                    if read:
                        h5py_copy(mtype[i], mspace_tmp[i], <void*> back_buf[i], <void*>progbuf[i], H5PY_GATHER)

            if read:
                if count > 1:
                    H5Dread_multi(count, <hid_t*> dset, <hid_t*>mtype, <hid_t*> mspace_tmp, <hid_t*>fspace_tmp, dxpl, conv_buf)
                else:
                    H5Dread(dset[0], dstype[0], cspace[0], fspace_tmp[0], dxpl, <void*> conv_buf[0])

                for i in range(count):
                    H5Tconvert(dstype[i], mtype[i], npoints[i], <void*> conv_buf[i], <void*> back_buf[i], dxpl)
                    h5py_copy(mtype[i], mspace_tmp[i], <void*> conv_buf[i], <void*>progbuf[i], H5PY_SCATTER)
            else:
                for i in range(count):
                    h5py_copy(mtype[i], mspace_tmp[i], <void*> conv_buf[i], <void*>progbuf[i], H5PY_GATHER)
                    H5Tconvert(mtype[i], dstype[i], npoints[i], <void*> conv_buf[i], <void*> back_buf[i], dxpl)

                if count > 1:
                    H5Dwrite_multi(count, <hid_t*>dset, <hid_t*>dstype, <hid_t*>cspace, <hid_t*>fspace_tmp, dxpl, <const void**> conv_buf)
                else:
                    H5Dwrite(dset[0], dstype[0], cspace[0], fspace_tmp[0], dxpl, <void*>  conv_buf[0])

                for i in range(count):
                    H5Dvlen_reclaim(dstype[i], cspace[i], H5P_DEFAULT, <void*>  conv_buf[i])

    finally:

        for i in range(count):
            if (back_buf != NULL) and (need_bkg[i]) and (back_buf[i] != NULL):
                    free(back_buf[i])

            if (conv_buf != NULL) and (conv_buf[i] != NULL):
                    free(conv_buf[i])

            if cspace and (cspace[i] > 0):
                H5Sclose(cspace[i])
            if dstype and (dstype[i] > 0):
                H5Tclose(dstype[i])

        if mspace_tmp != NULL:
            free(mspace_tmp)
        if fspace_tmp != NULL:
            free(fspace_tmp)

        if npoints != NULL:
            free(npoints)
        if need_bkg != NULL:
            free(need_bkg)
        if back_buf != NULL:
            free(back_buf)
        if conv_buf != NULL:
            free(conv_buf)
        if cspace != NULL:
            free(cspace)

        if rawdstype > 0:
            H5Tclose(rawdstype)

cdef hid_t make_reduced_type(hid_t mtype, hid_t dstype):
    # Go through dstype, pick out the fields which also appear in mtype, and
    # return a new compound type with the fields packed together
    # See also: issue 372

    cdef hid_t newtype, temptype
    cdef hsize_t newtype_size, offset
    cdef char* member_name = NULL
    cdef int idx

    # Make a list of all names in the memory type.
    mtype_fields = []
    for idx in range(H5Tget_nmembers(mtype)):
        member_name = H5Tget_member_name(mtype, idx)
        try:
            mtype_fields.append(member_name)
        finally:
            H5free_memory(member_name)
            member_name = NULL

    # First pass: add up the sizes of matching fields so we know how large a
    # type to make
    newtype_size = 0
    for idx in range(H5Tget_nmembers(dstype)):
        member_name = H5Tget_member_name(dstype, idx)
        try:
            if member_name not in mtype_fields:
                continue
            temptype = H5Tget_member_type(dstype, idx)
            newtype_size += H5Tget_size(temptype)
            H5Tclose(temptype)
        finally:
            H5free_memory(member_name)
            member_name = NULL

    newtype = H5Tcreate(H5T_COMPOUND, newtype_size)

    # Second pass: pick out the matching fields and pack them in the new type
    offset = 0
    for idx in range(H5Tget_nmembers(dstype)):
        member_name = H5Tget_member_name(dstype, idx)
        try:
            if member_name not in mtype_fields:
                continue
            temptype = H5Tget_member_type(dstype, idx)
            H5Tinsert(newtype, member_name, offset, temptype)
            offset += H5Tget_size(temptype)
            H5Tclose(temptype)
        finally:
            H5free_memory(member_name)
            member_name = NULL

    return newtype


cdef void* create_buffer(size_t ipt_size, size_t opt_size, size_t nl) except NULL:

    cdef size_t final_size
    cdef void* buf

    if ipt_size >= opt_size:
        final_size = ipt_size*nl
    else:
        final_size = opt_size*nl

    buf = malloc(final_size)
    if buf == NULL:
        raise MemoryError("Failed to allocate conversion buffer")

    return buf

# =============================================================================
# Scatter/gather routines

ctypedef struct h5py_scatter_t:
    size_t i
    size_t elsize
    void* buf

cdef herr_t h5py_scatter_cb(void* elem, hid_t type_id, unsigned ndim,
                const hsize_t *point, void *operator_data) except -1 nogil:
    cdef h5py_scatter_t* info = <h5py_scatter_t*>operator_data

    memcpy(elem, (<char*>info[0].buf)+((info[0].i)*(info[0].elsize)),
           info[0].elsize)

    info[0].i += 1

    return 0

cdef herr_t h5py_gather_cb(void* elem, hid_t type_id, unsigned ndim,
                const hsize_t *point, void *operator_data) except -1 nogil:
    cdef h5py_scatter_t* info = <h5py_scatter_t*>operator_data

    memcpy((<char*>info[0].buf)+((info[0].i)*(info[0].elsize)), elem,
            info[0].elsize)

    info[0].i += 1

    return 0

# Copy between a contiguous and non-contiguous buffer, with the layout
# of the latter specified by a dataspace selection.
cdef herr_t h5py_copy(hid_t tid, hid_t space, void* contig, void* noncontig,
                 copy_dir op) except -1:

    cdef h5py_scatter_t info
    cdef hsize_t elsize

    elsize = H5Tget_size(tid)

    info.i = 0
    info.elsize = elsize
    info.buf = contig

    if op == H5PY_SCATTER:
        H5Diterate(noncontig, tid, space, h5py_scatter_cb, &info)
    elif op == H5PY_GATHER:
        H5Diterate(noncontig, tid, space, h5py_gather_cb, &info)
    else:
        raise RuntimeError("Illegal direction")

    return 0

# =============================================================================
# VLEN support routines

cdef htri_t needs_bkg_buffer(hid_t src, hid_t dst) except -1:

    cdef H5T_cdata_t *info = NULL

    if H5Tdetect_class(src, H5T_COMPOUND) or H5Tdetect_class(dst, H5T_COMPOUND):
        return 1

    try:
        H5Tfind(src, dst, &info)
    except:
        print("Failed to find converter for %s -> %s" % (H5Tget_size(src), H5Tget_tag(dst)))
        raise

    if info[0].need_bkg == H5T_BKG_YES:
        return 1

    return 0

# Determine if the given type requires proxy buffering
cdef htri_t needs_proxy(hid_t tid) except -1:

    cdef H5T_class_t cls
    cdef hid_t supertype
    cdef int i, n
    cdef htri_t result

    cls = H5Tget_class(tid)

    if cls == H5T_VLEN or cls == H5T_REFERENCE:
        return 1

    elif cls == H5T_STRING:
        return H5Tis_variable_str(tid)

    elif cls == H5T_ARRAY:

        supertype = H5Tget_super(tid)
        try:
            return needs_proxy(supertype)
        finally:
            H5Tclose(supertype)

    elif cls == H5T_COMPOUND:

        n = H5Tget_nmembers(tid)
        for i in range(n):
            supertype = H5Tget_member_type(tid, i)
            try:
                result = needs_proxy(supertype)
                if result > 0:
                    return 1
            finally:
                H5Tclose(supertype)
        return 0

    return 0
