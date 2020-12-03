"""A simple example of building a virtual dataset from dataset sub-selections.

create a file with two datasets "original_data/d1" and "original_data/d2".
Both are numpy 3D arrays

We want to create a virtual dataset from slices 2 and 3 of "d1" and
"7, 8, 9" of "d2 to create
"""

import h5py
import numpy.random
import h5py._hl.selections as selection


file_path = "output_vds_advance.hdf5"


def create_datasets():
    """create original dataset"""
    with h5py.File(file_path, mode="w") as h5s:
        h5s["original_data/d1"] = numpy.arange(10*10*10).reshape(10, 10, 10)
        h5s["original_data/d2"] = numpy.arange(10*10*10, 10*10*10*2).reshape(10, 10, 10)


def create_vds():
    """create virtual dataset"""
    # fancy selection using a list
    v_source_1 = h5py.VirtualSource(file_path, "original_data/d1",
                                    shape=(10, 10, 10))
    v_source_1 = v_source_1[[2, 3]]

    # fancy selection using a slice
    v_source_2 = h5py.VirtualSource(file_path, "original_data/d2",
                                    shape=(10, 10, 10))
    v_source_2 = v_source_2[slice(7, 10)]

    with h5py.File(file_path, mode="a") as h5s:
        layout = h5py.VirtualLayout(shape=(5, 10, 10), dtype="i4")
        layout[0:2] = v_source_1
        layout[2:] = v_source_2

        h5s.create_virtual_dataset("vds", layout, fillvalue=-5)


create_datasets()
create_vds()
