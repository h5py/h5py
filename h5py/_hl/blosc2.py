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

import numpy

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
        and dataset.dtype.byteorder in ('=', '|')
        and (dataset.file.mode == 'r'
             or platform.system().lower() != 'windows')
    )

def opt_slicing_enabled():
    """Is Blosc2 optimized slicing not disabled via the environment?"""
    # The BLOSC2_FILTER environment variable set to a non-zero integer
    # forces the use of the filter pipeline.
    try:
        force_filter = int(os.environ.get('BLOSC2_FILTER', '0'), 10)
    except ValueError:
        force_filter = 0
    return force_filter == 0

def _read_chunk_slice(path, offset, slice_, _dtype):  # TODO: drop _dtype
    # TODO: implement
    shape = tuple(s.stop - s.start for s in slice_)
    return numpy.arange(numpy.product(shape), dtype=_dtype).reshape(shape)

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
    print("XXXX B2NDopt slice:", slice_)  # TODO: remove
    slice_arr = numpy.empty(dtype=dataset.dtype, shape=slice_shape)

    # TODO: consider using 'dataset.id.get_chunk_info' for performance
    get_chunk_info = dataset.id.get_chunk_info_by_coord
    for chunk_slice in dataset.iter_chunks(slice_):
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
        print(f"XXXX B2NDopt chunk slice: {chunk_slice} (<-{slice_as_chunk_slice}) -> {chunk_as_slice_slice}")  # TODO: remove
        chunk_info = get_chunk_info(chunk_slice_start)
        print("XXXX B2NDopt chunk_info:", chunk_info)  # TODO: remove
        chunk_slice_arr = _read_chunk_slice(dataset.file.filename, chunk_info.byte_offset,
                                            slice_as_chunk_slice, _dtype=dataset.dtype)
        if (chunk_slice_arr.dtype != dataset.dtype
            or len(chunk_slice_arr.shape) != len(slice_shape)
            or chunk_slice_arr.shape > slice_shape):
            raise RuntimeError(f"Invalid shape/dtype of chunk covering coordinate {chunk_slice_start} "
                               f"(offset {chunk_info.byte_offset}): "
                               f"expected <= {slice_shape}/{dataset.dtype}, "
                               f"got {chunk_slice_arr.shape}/{chunk_slice_arr.dtype}")

        slice_arr[chunk_as_slice_slice] = chunk_slice_arr
    return slice_arr
