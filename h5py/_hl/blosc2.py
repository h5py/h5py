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

    If you have checked these conditions manually, you may use
    `opt_selection_read()`.

    If a dataset is adapted for Blosc2 optimized slicing, you may just use
    `opt_slice_read()`, which takes care of the checks.
"""

import os
import platform

import numpy

try:
    # We only want to automatically enable
    # both optimized and filter-based slicing
    # if both Python-Blosc2 and hdf5plugin are available.
    #
    # If only Python-Blosc2 is available,
    # enabling optimized slicing only would be quite confusing
    # as some slice operations would work while others would fail.
    # If only hdf5plugin is available,
    # enabling filter-based slicing only would enable all the plugins
    # without the user's knowledge, just because the package is available.
    #
    # In other words, the user needs to import hdf5plugin explicitly
    # to read Blosc2-compressed data in the absence of Python-Blosc2.
    #
    # This means that the order of imports is relevant here.
    from blosc2.schunk import open as blosc2_schunk_open
    import hdf5plugin
except ImportError:
    blosc2_schunk_open = None

from . import selections as sel


class NoOptSlicingError(TypeError):
    """Blosc2 optimized slicing is not possible."""
    pass


def opt_slicing_selection_ok(selection):
    """Is the given selection suitable for Blosc2 optimized slicing?"""
    return (
        isinstance(selection, sel.SimpleSelection)
        and numpy.prod(selection._sel[2]) == 1  # all steps equal 1
    )


def opt_slicing_dataset_ok(dataset):
    """Is the given dataset suitable for Blosc2 optimized slicing?

    It is assumed that the dataset is also ok for fast reading.  The result
    may be cached.
    """
    return (
        dataset.chunks is not None
        # '.compression' and '.compression_opts' don't work with plugins:
        # <https://forum.hdfgroup.org/t/registering-custom-filter-issues/9239>
        and '32026' in dataset._filters  # Blosc2's ID
        and dataset.dtype.isnative
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
    # hdf5-blosc2 always uses an opaque dtype, convert the array
    # (the wrapping below does not copy the data anyway).
    return numpy.ndarray(s.shape, dtype=dtype, buffer=s.data)


def opt_selection_read(dataset, selection, new_dtype=None):
    """Read the specified selection from the given dataset.

    Blosc2 optimized slice reading is used, but the caller must make sure
    beforehand that both the dataset and the selection are suitable for such
    operation.

    A NumPy array is returned with the desired slice.  The array will have the
    given new dtype if specified.
    """
    slice_start = selection._sel[0]
    slice_shape = selection.mshape
    slice_ = tuple(slice(st, st + sh)
                   for (st, sh) in zip(slice_start, slice_shape))
    slice_arr = numpy.empty(dtype=new_dtype or dataset.dtype,
                            shape=slice_shape)
    if 0 in slice_shape:  # empty slice
        return slice_arr.reshape(selection.array_shape)

    # TODO: consider using 'dataset.id.get_chunk_info' for performance
    get_chunk_info = dataset.id.get_chunk_info_by_coord
    for chunk_slice in dataset.iter_chunks(slice_):
        # TODO: Remove when #2341 is fixed.
        if any(s.stop <= s.start for s in chunk_slice):
            continue  # bogus iter_chunks item, see #2341

        # Compute different parameters for the slice/chunk combination.
        (
            slice_as_chunk_slice,
            chunk_as_slice_slice,
            chunk_slice_start,
        ) = tuple(zip(*(
            (  # nth value below gets added to nth tuple above
                slice(csl.start % csh, ((csl.start % csh)
                                        + (csl.stop - csl.start))),
                slice(csl.start - sst, csl.stop - sst),
                csl.start,
            )
            for (csl, csh, sst)
            in zip(chunk_slice, dataset.chunks, slice_start)
        )))

        # Get the part of the slice that overlaps the current chunk.
        chunk_info = get_chunk_info(chunk_slice_start)
        chunk_slice_arr = _read_chunk_slice(
            dataset.file.filename, chunk_info.byte_offset,
            slice_as_chunk_slice, dataset.dtype)
        if (
                chunk_slice_arr.dtype != dataset.dtype
                or len(chunk_slice_arr.shape) != len(slice_shape)
                or chunk_slice_arr.shape > slice_shape
        ):
            # The data in the Blosc2 super-chunk is bogus.
            raise RuntimeError(
                f"Invalid shape/dtype of "
                f"chunk covering coordinate {chunk_slice_start} "
                f"(offset {chunk_info.byte_offset}): "
                f"expected <= {slice_shape}/{dataset.dtype}, "
                f"got {chunk_slice_arr.shape}/{chunk_slice_arr.dtype}")

        # Place the part in the final slice.
        slice_arr[chunk_as_slice_slice] = chunk_slice_arr

    # Adjust result dimensions to those dictated by the input selection.
    ret_shape = selection.array_shape
    if ret_shape == ():  # scalar result
        return slice_arr[()]
    return slice_arr.reshape(ret_shape)


def opt_slice_read(dataset, slice_, new_dtype=None):
    """Read the specified slice from the given dataset.

    The dataset must support a ``_blosc2_opt_slicing_ok`` property that calls
    `opt_slicing_dataset_ok()`.

    Blosc2 optimized slice reading is used if available and suitable,
    otherwise a `NoOptSlicingError` is raised.

    A NumPy array is returned with the desired slice.  The array will have the
    given new dtype if specified.
    """
    if not dataset._blosc2_opt_slicing_ok:
        raise NoOptSlicingError(
            "Dataset is not suitable for Blosc2 optimized slicing")

    if not opt_slicing_enabled():
        raise NoOptSlicingError(
            "Blosc2 optimized slicing is unavailable or disabled")

    selection = sel.select(dataset.shape, slice_, dataset=dataset)
    if not opt_slicing_selection_ok(selection):
        raise NoOptSlicingError(
            "Selection is not suitable for Blosc2 optimized slicing")

    return opt_selection_read(dataset, selection, new_dtype)
