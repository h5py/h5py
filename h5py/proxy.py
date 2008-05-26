import tempfile
import os
import numpy

import h5d, h5s, h5t, h5f, h5p, h5z

class ProxyError(StandardError):
    pass

class DatasetProxy(object):

    """
        Thin class which acts as an interface to an HDF5 dataset object.

    """
    def begin_proxy(self):


        if self.proxy_id is not None:
            raise ProxyError("Already proxying.")

        fid = 0
        space_id = 0
        plist_id = 0
        type_id = 0
        proxy_id = 0
        fname = tempfile.mktemp('.hdf5')

        try:
            space_id = h5d.get_space(self.id)
            type_id = h5g.get_type(self.id)
            plist_id = h5g.get_create_plist(self.id)

            h5p.remove_filter(plist_id, h5z.FILTER_ALL)
            h5p.set_alloc_time(plist_id, h5p.ALLOC_TIME_INCR)

            fid = h5f.create(fname, h5f.ACC_RDWR)
            proxy_id = h5d.create(fid, "PROXY", type_id, space_id, plist_id)
        except:
            if fid != 0:
                h5f.close(fid)
            if space_id != 0:
                h5s.close(space_id)
            raise
        finally:
            if plist_id != 0:
                h5p.close(plist_id)
            if type_id != 0:
                h5t.close(type_id)

        self._proxy_fid = fid
        self._proxy_fname = fname
        self._proxy_space = space_id
        self._proxy_id = proxy_id

    def end_proxy(self):

        if not hasattr(self, '_proxy_id') or self._proxy_id is None:
            raise ProxyError("Not proxying.")

        h5s.close(self._proxy_space)
        h5d.close(self._proxy_id)
        h5f.close(self._proxy_fid)
        self._proxy_id = None
        os.unlink(self._proxy_fname)

    def _read(self, start, count, stride=None, **kwds):
        """ Dataset read access.  In direct mode, simply reads data from 
            self.id.  In proxy mode, reads unmodified data from self.id and
            modified sections from self._proxy_id)

            Don't call this directly.
        """
        if self.proxy_id is None:
            return h5d.py_read_slab(self.id, start, count, stride, **kwds)

        else:
            mem_space = 0
            backing_space = 0
            patch_space = 0
            
            try:
                mem_space = h5s.create_simple(count)    

                # Create Numpy array
                dtype = h5t.py_dtype(self._proxy_id)
                arr = numpy.ndarray(count, dtype=dtype)

                patch_space = h5s.copy(self._proxy_space)
                backing_space = h5s.copy(self._proxy_space)

                # What needs to be read from the original dataset.
                # This is all elements of the new selection which are not
                # already selected in self._proxy_space
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
                    h5d.read(self._proxy_id, mem_space, patch_space, arr)

            finally:
                if mem_space != 0:
                    h5s.close(mem_space)
                if backing_space != 0:
                    h5s.close(backing_space)
                if patch_space != 0:
                    h5s.close(patch_space)

            return arr

    def _write(self, arr, start, stride=None):
        
        if self.proxy_id is None:
            h5d.py_write_slab(self.id, arr, start, stride)
        
        else:
            # We get free argument validation courtesy of this function.
            h5d.py_write_slab(self._proxy_id, arr, start, stride)

            # Record this section of the dataspace as changed.
            count = arr.shape
            h5s.select_hyperslab(self._proxy_space, start, count, stride, op=h5s.SELECT_OR)

    def commit(self):

        h5d.py_patch(self._proxy_id, self.id, self._proxy_space)
        h5s.select_none(self._proxy_space)

    def rollback(self):

        # Proxy file doesn't shrink, but space will be re-used.
        # Worst case == proxy file is size of the original dataset, sans
        # compression
        h5s.select_none(self._proxy_space)
            
        

















            
