from copy import deepcopy as copy
from collections import namedtuple
from .. import h5s, h5t, h5
from .. import version


class VDSmap(namedtuple('VDSmap', ('vspace', 'file_name',
                                   'dset_name', 'src_space'))):
    '''Mapping to the the virtual data set from a source dataset

    P
    '''


vds_support = False
hdf5_version = version.hdf5_version_tuple[0:3]

if hdf5_version >= h5.get_config().vds_min_hdf5_version:
    vds_support = True


class DatasetContainer(object):
    def __init__(self, path, key, shape, dtype=None, maxshape=None):
        """Base class for virtual dataset components.

        This is an object that looks like a dataset, but it not. It
        allows the user to specify the maps based on lazy indexing,
        which is natural, but without needing to load the data.

        path
            This is the full path to the file on disk.  This must be
            text-like or Path-like

        key
            This is the key to the entry inside the hdf5 file.  Must be
            text-like

        shape
            The shape of the data.

            We specify this by hand because it is a lot faster than
            getting it from the source file.

        dtype
            The data type. For the source we specify this because it
            is faster than getting from the file. For the target, we
            can specify it to be different to the source.

        """
        self.path = path
        self.key = key
        self.shape = shape
        # if we don't slice, we want the whole array
        self.slice_list = [slice(0, ix, 1) for ix in self.shape]

        if maxshape is None:
            self.maxshape = shape
        else:
            self.maxshape = tuple([h5s.UNLIMITED if ix is None else ix
                                   for ix in maxshape])

    @property
    def key(self):
        return self._key_bytes

    @key.setter
    def key(self, val):
        self._key_bytes = val.encode('utf-8')

    @property
    def path(self):
        return self._path_bytes

    @path.setter
    def path(self, val):
        self._path_bytes = val.encode('utf-8')

    def _parse_slicing(self, key):
        """
        parses the __get_item__ key to get useful slicing information
        """
        tmp = copy(self)
        rank = len(self.shape)
        if (rank-len(key)) < 0:
            raise IndexError('Index rank is greater than dataset rank')
        if isinstance(key[0], tuple):   # sometimes this is needed. odd
            key = key[0]
        key = list(key)
        key = [slice(ix, ix + 1, 1) if isinstance(ix, (int, float)) else ix
               for ix in key]

        # now let's parse ellipsis
        ellipsis_test = [ix is Ellipsis for ix in key]
        if sum(ellipsis_test) > 1:
            raise ValueError("Only use of one Ellipsis(...) supported.")
        if not any(ellipsis_test):
            tmp.slice_list[:len(key)] = key
        elif any(ellipsis_test) and (len(key) is not 1):
            ellipsis_idx = ellipsis_test.index(True)
            ellipsis_idx_back = ellipsis_test[::-1].index(True)
            tmp.slice_list[0:ellipsis_idx] = key[0:ellipsis_idx]
            if ellipsis_idx_back >= ellipsis_idx:  # edge case
                tmp.slice_list[-ellipsis_idx_back:] = key[-ellipsis_idx_back:]

        new_shape = []
        for ix, sl in enumerate(tmp.slice_list):
            step = 1 if sl.step is None else sl.step
            if step > 0:
                start = 0 if sl.start is None else sl.start  # parse for Nones
                stop = self.shape[ix] if sl.stop is None else sl.stop
                start = self.shape[ix]+start if start < 0 else start
                stop = self.shape[ix]+stop if stop < 0 else stop
                if start < stop:
                    new_shape.append((stop - start + step - 1)//step)
                else:
                    new_shape.append(0)

            elif step < 0:
                stop = 0 if sl.stop is None else sl.stop  # parse for Nones
                start = self.shape[ix] if sl.start is None else sl.start

                start = self.shape[ix]+start if start < 0 else start
                stop = self.shape[ix]+stop if stop < 0 else stop

                if start > stop:  # this gets the same behaviour as numpy array
                    new_shape.append((start - stop - step - 1)//-step)
                else:
                    new_shape.append(0)
            elif step == 0:
                raise IndexError("A step of 0 is not valid")
            tmp.slice_list[ix] = slice(start, stop, step)
        tmp.shape = tuple(new_shape)
        return tmp


class VirtualSource(DatasetContainer):
    """VDS 'source' object

    A container for the source information. This is similar to a
    virtual target, but the shape information changes with slicing.
    This does not happen with VirtualTarget since it is the source
    that ultimately set's the block shape.

    """
    def __getitem__(self, *key):
        return self._parse_slicing(key)


class VirtualTarget(DatasetContainer):
    """VDS 'target' object

    A container for the target information. This is similar to a
    virtual source, but the shape information does not change with
    slicing.  This does not happen with VirtualSource since it is the
    source that ultimately set's the block shape so it must change on
    slicing.

    """
    def __getitem__(self, *key):
        tmp = self._parse_slicing(key)
        tmp.shape = self.shape
        return tmp


class VirtualMap(object):
    def __init__(self, virtual_source, virtual_target, dtype):
        """Maintain the mapping between the source and target datasets

        The idea of this class is to specify the mapping between the
        source and target files.  Since data type casting is supported
        by VDS, we include this here.

            virtual_source
                A DatasetContainer object containing all the useful
                information about the source file for this map.

            virtual_target
                A DatasetContainer object containing all the useful
                information about the source file for this map.

            dtype
                The type of the final output dataset.

        """
        self.src = virtual_source[...]
        self.dtype = dtype
        self.target = virtual_target[...]
        self.block_shape = None

        # if the rank of the two datasets is not the same,
        # left-pad with size 1 dimensions
        # This isn't necessarily the best way to do this!
        rank_def = len(self.target.shape) - len(self.src.shape)
        if rank_def > 0:
            if len(self.src.shape) == 1:
                pass
            else:
                self.block_shape = (1,)*rank_def + self.src.shape
        elif rank_def < 0:
            # This might be pathological.
            if len(self.target.shape) == 1:
                pass
            else:
                self.block_shape = (1,)*rank_def + self.target.shape
        else:
            self.block_shape = self.src.shape

        self.src_dspace = h5s.create_simple(self.src.shape, self.src.maxshape)

        start_idx = tuple([ix.start for ix in self.src.slice_list])
        stride_idx = tuple([ix.step for ix in self.src.slice_list])
        count_idx = tuple([ix.stop - ix.start for ix in self.src.slice_list])

        if any(ix == h5s.UNLIMITED for ix in self.src.maxshape):
            unlimited_index = self.src.maxshape.index(h5s.UNLIMITED)
            count_idx = list(count_idx)
            count_idx[unlimited_index] = h5s.UNLIMITED
            count_idx = tuple(count_idx)
            bs = list(self.block_shape)
            bs[unlimited_index] = 1
            self.block_shape = tuple(bs)

        self.src_dspace.select_hyperslab(
            start=start_idx, count=count_idx, stride=stride_idx,
            block=self.block_shape)

    def get_src_dpsace(self):
        '''
        get the source data space
        '''
        return self.src_dspace


def vmlist_to_kwawrgs(VMlist, fillvalue):
    '''Create kwargs to pass to Group.create_virtual_dataset

    Parameters
    ----------
    VMlist : List[VirtualMap]

    fillvalue : object

    Returns
    -------
    kwargs : dict
       Suitable to be **kwarg into `Group.create_virtual_dataset`
    '''
    if not VMlist:
        raise ValueError("create_virtual_dataset requires at least "
                         "one virtual map to construct output.")

    if not isinstance(VMlist, (tuple, list)):
        VMlist = [VMlist]

    vds_iter = []

    sh = VMlist[0].target.shape
    max_sh = VMlist[0].target.maxshape
    for VM in VMlist:
        virt_start_idx = tuple([ix.start
                                for ix in VM.target.slice_list])
        virt_count_idx = tuple([ix.stop - ix.start
                                for ix in VM.target.slice_list])
        virt_stride_index = tuple([ix.step
                                   for ix in VM.target.slice_list])

        if any(ix == h5s.UNLIMITED for ix in VM.target.maxshape):
            unlimited_index = VM.target.maxshape.index(h5s.UNLIMITED)
            virt_count_idx = list(virt_count_idx)
            virt_count_idx[unlimited_index] = h5s.UNLIMITED
            virt_count_idx = tuple(virt_count_idx)

        virt_dspace = h5s.create_simple(sh, max_sh)
        virt_dspace.select_hyperslab(start=virt_start_idx,
                                     count=virt_count_idx,
                                     stride=virt_stride_index,
                                     block=VM.block_shape)
        vds_iter.append(VDSmap(virt_dspace,
                               VM.src.path,
                               VM.src.key,
                               VM.src_dspace))

    return {
        'name': VMlist[-1].target.key,
        'vds_iter': vds_iter,
        'target_dtype': h5t.py_create(VMlist[-1].dtype, logical=1),
        'target_shape': VMlist[0].target.shape,
        'target_maxshape': VMlist[0].target.maxshape,
        'fillvalue': fillvalue}
