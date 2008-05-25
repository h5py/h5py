import tempfile
import os
import numpy

import h5d, h5s, h5t, h5f

class ProxyError(StandardError):
    pass

class DatasetProxy(object):

    """
        Thin class which acts as an interface to an HDF5 dataset object.

    """
    def begin_proxy(self):

        # todo: modify plist to enforce late allocation and no compression

        if self.proxy_id is not None:
            raise ProxyError("Already proxying.")

        fid = 0
        sid = 0
        pid = 0
        tid = 0
        proxy_id = 0
        fname = tempfile.mktemp('.hdf5')

        try:
            sid = h5d.get_space(self.id)
            pid = h5g.get_create_plist(self.id)
            tid = h5g.get_type(self.id)

            fid = h5f.create(fname, h5f.ACC_RDWR)
            proxy_id = h5d.create(fid, "PROXY", tid, sid, pid)
        except:
            if fid != 0:
                h5f.close(fid)
            if sid != 0:
                h5s.close(sid)
            raise
        finally:
            if pid != 0:
                h5p.close(pid)
            if tid != 0:
                h5t.close(tid)

        self.fid = fid
        self.space_id = sid
        self.proxy_id = proxy_id
        self.fname = fname

    def end_proxy(self):

        if self.proxy_id is None:
            raise ProxyError("Not proxying.")

        h5s.close(self.space_id)
        h5d.close(self.proxy_id)
        h5f.close(self.fid)
        os.unlink(self.fname)
        self.proxy_id = None


    def read(self, start, count, stride=None, **kwds):

        # todo: argument validation

        if self.proxy_id is None:
            return h5d.py_read_slab(self.id, start, count, stride, **kwds)

        else:
            mem_space = 0
            backing_space = 0
            patch_space = 0
            tid = 0
            
            try:
                mem_space = h5s.create_simple(count)    

                # Create Numpy array
                tid = h5d.get_type(self.proxy_id)
                dtype = h5t.py_h5t_to_dtype(tid, **kwds)
                arr = numpy.ndarray(count, dtype=dtype)

                patch_space = h5s.copy(self.space_id)
                backing_space = h5s.copy(self.space_id)

                # What needs to be read from the original dataset.
                # This is all elements of the new selection which are not
                # marked as modified.
                h5s.select_hyperslab(backing_space, start, count, stride, op=h5s.SELECT_NOTA)

                # What needs to be read from the proxy dataset.
                # This is the intersection of the modified selection and the
                # requested selection.
                h5s.select_hyperslab(patch_space, start, count, stride, op=h5s.SELECT_AND)

                # Read from the original dataset.
                if h5s.get_select_npoints(backing_space) > 0:
                    h5d.read(self.id, mem_space, backing_space, arr)

                # Read the rest from the proxy dataset.
                if h5s.get_select_npoints(patch_space) > 0:
                    h5d.read(self.proxy_id, mem_space, patch_space, arr)

            finally:
                if mem_space != 0:
                    h5s.close(mem_space)
                if backing_space != 0:
                    h5s.close(backing_space)
                if patch_space != 0:
                    h5s.close(patch_space)
                if tid != 0:
                    h5t.close(tid)

            return arr

    def write(self, arr, start, stride=None):
        
        if self.proxy_id is None:
            h5d.py_write_slab(self.id, arr, start, stride)
        
        else:
            # We get free argument validation courtesy of this function.
            h5d.py_write_slab(self.proxy_id, arr, start, stride)

            # Record this section of the dataspace as changed.
            count = arr.shape
            h5s.select_hyperslab(self.space_id, start, count, stride, op=h5s.SELECT_OR)

    def commit(self):

        # this will use the yet-unwritten h5d.py_patch function
        pass

    def rollback(self):

        # fixme: this leaks file space
        h5s.select_none(self.space_id)
            
        

















            
