# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2023 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements support for Blosc2 optimized slicing.

    Please note that for a selection over a dataset to be suitable for Blosc2
    optimized slicing, besides being amenable to fast reading, (i) such
    slicing must be enabled globally (`opt_slicing_enabled()`), (ii) the
    dataset must be amenable to it (`opt_slicing_dataset_ok()`), and (iii) the
    selection must be amenable to it (`opt_slicing_selection_ok()`).
"""

import os
import platform
import sys

import numpy

try:
    from blosc2.schunk import open as blosc2_schunk_open
except ImportError:
    blosc2_schunk_open = None

from . import selections as sel


def opt_slicing_selection_ok(selection):
    """Is the given selection suitable for Blosc2 optimized slicing?"""
    return (isinstance(selection, sel.SimpleSelection)
            and numpy.prod(selection._sel[2]) == 1  # all steps equal 1
    )

def opt_slicing_dataset_ok(dataset):
    """Is the given dataset suitable for Blosc2 optimized slicing?

    It is assumed that the dataset is also ok for fast reading.  The result
    may be cached.
    """
    return (
        dataset.chunks is not None
        # '.compression' and '.compression_opts' don't work with plugins,
        # see <https://forum.hdfgroup.org/t/registering-custom-filter-issues/9239>.
        and '32026' in dataset._filters  # Blosc2's ID
        and (dataset.dtype.byteorder
             in ('=', '|', dict(little='<', big='>')[sys.byteorder]))
        and (dataset.file.mode == 'r'
             or platform.system().lower() != 'windows')
    )

def opt_slicing_enabled():
    """Is Blosc2 optimized slicing not disabled via the environment?

    This returns false if Blosc2 is not usable or if the BLOSC2_FILTER
    environment variable is set to a non-zero integer (which forces the use of
    the HDF5 filter pipeline).
    """
    if blosc2_schunk_open is None:
        return False
    try:
        force_filter = int(os.environ.get('BLOSC2_FILTER', '0'), 10)
    except ValueError:
        force_filter = 0
    return force_filter == 0

def _read_chunk_slice(path, offset, slice_, dtype):
    schunk = blosc2_schunk_open(path, mode='r', offset=offset)
    s = schunk[slice_]
    if s.dtype.kind != 'V':
        return s
    # hdf5-blosc2 always uses an opaque dtype, convert the array.
    return numpy.ndarray(s.shape, dtype=dtype, buffer=s.data)

def opt_slice_read(dataset, selection):
    """Read the specified selection from the given dataset.

    Blosc2 optimized slice reading is used, but the caller must make sure
    beforehand that both the dataset and the selection are suitable for such
    operation.

    A NumPy array is returned with the desired slice.
    """
    slice_start = selection._sel[0]
    slice_shape = selection.mshape
    slice_ = tuple(slice(st, st + sh)
                   for (st, sh) in zip(slice_start, slice_shape))
    slice_arr = numpy.empty(dtype=dataset.dtype, shape=slice_shape)

    # TODO: consider using 'dataset.id.get_chunk_info' for performance
    get_chunk_info = dataset.id.get_chunk_info_by_coord
    for chunk_slice in dataset.iter_chunks(slice_):
        # Compute different parameters for the slice/chunk combination.
        (
            slice_as_chunk_slice,
            chunk_as_slice_slice,
            chunk_slice_start,
        ) = tuple(zip(*(
            (  # nth value below gets added to nth tuple above
                slice(csl.start % csh, (csl.start % csh) + (csl.stop - csl.start)),
                slice(csl.start - sst, csl.stop - sst),
                csl.start,
            )
            for (csl, csh, sst)
            in zip(chunk_slice, dataset.chunks, slice_start)
        )))

        # Get the part of the slice that overlaps the current chunk.
        chunk_info = get_chunk_info(chunk_slice_start)
        chunk_slice_arr = _read_chunk_slice(dataset.file.filename, chunk_info.byte_offset,
                                            slice_as_chunk_slice, dataset.dtype)
        if (chunk_slice_arr.dtype != dataset.dtype
            or len(chunk_slice_arr.shape) != len(slice_shape)
            or chunk_slice_arr.shape > slice_shape):
            raise RuntimeError(f"Invalid shape/dtype of chunk covering coordinate {chunk_slice_start} "
                               f"(offset {chunk_info.byte_offset}): "
                               f"expected <= {slice_shape}/{dataset.dtype}, "
                               f"got {chunk_slice_arr.shape}/{chunk_slice_arr.dtype}")

        # Place the part in the final slice.
        slice_arr[chunk_as_slice_slice] = chunk_slice_arr

    return slice_arr
