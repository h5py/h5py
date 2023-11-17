# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2023 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""Example of Blosc2 NDim optimized slicing.

It creates a 2-dimensional dataset made of different chunks, compressed with
Blosc2.  Then it proceeds to slice the dataset in ways that may and may not
benefit from Blosc2 optimized slicing.  Some hints about forcing the use of
the HDF5 filter pipeline are included, as well as comments on the Python
package dependencies required for the different use cases.

Optimized slicing can provide considerable speed-ups in certain use cases,
please see `this benchmark`__ which evaluates applying the same technique in
PyTables, and the post `Optimized Hyper-slicing in PyTables with Blosc2
NDim`_, which presents the results of the benchmark.

__ https://github.com/PyTables/PyTables/blob/master/bench/b2nd_compare_getslice.py

.. _Optimized Hyper-slicing in PyTables with Blosc2 NDim:
   https://www.blosc.org/posts/pytables-b2nd-slicing/
"""

import os

import h5py
import numpy as np


# The array created below is to be stored in a dataset
# 3 chunks top-down, 3 chunks accross,
# with fringe chunks partially filled by array data::
#
#      <- 100 -X- 100 -X-50->···
#    ^ +-------+-------+----+··+
#    1 |       |       |    |  ·
#    0 |  #0   |  #1   | #2 |  ·
#    0 |       |       |    |  ·
#    X +-------+-------+----+··+
#    1 |       |       |    |  ·
#    0 |  #3   |  #4   | #5 |  ·
#    0 |       |       |    |  ·
#    X +-------+-------+----+··+
#    5 |       |       |    |  ·
#    0 |  #6   |  #7   | #8 |  ·
#    v +-------+-------+----+  ·
#    · +·······+·······+·······+
#
shape = (250, 250)
chunks = (100, 100)
data = np.arange(np.prod(shape)).reshape(shape)

file_name = 'b2nd-example.h5'
dataset_name = 'data'

# Creating a Blosc2-compressed dataset
# ------------------------------------
with h5py.File(file_name, 'w') as f:
    # This import is needed to declare Blosc2 compression parameters
    # for a newly created dataset.
    # For the moment, all writes to Blosc2-compressed datasets
    # use the HDF5 filter pipeline, so only hdf5plugin is needed.
    # If Python-Blosc2 is not available on your system,
    # importing hdf5plugin is enough to store your data.
    import hdf5plugin as h5p
    comp = h5p.Blosc2(cname='lz4', clevel=5, filters=h5p.Blosc2.SHUFFLE)
    dataset = f.create_dataset(dataset_name, data=data, **comp)

# Benefitting from Blosc2 optimized slicing
# -----------------------------------------
# Support for Blosc2 optimized slicing
# depends on *both* Python-Blosc2 and hdf5plugin.
# If they are available, the feature is enabled by default
# unless disabled via the `BLOSC2_FILTER` environment variable.
with h5py.File(file_name, 'r') as f:
    # If support for Blosc2 optimized slicing is available,
    # there is no need to import anything else explicitly for reading.
    # One just uses slicing as usual.
    dataset = f[dataset_name]
    # Slices with step == 1 may be optimized.
    slice_ = dataset[150:, 150:]
    print("Contiguous slice from dataset:", slice_, sep='\n')
    print("Contiguous slice from input array:", data[150:, 150:], sep='\n')
    # Slices with step != 1 (or with datasets of a foreign endianness)
    # are not optimized, but still work
    # (via the HDF5 filter pipeline and hdf5plugin).
    slice_ = dataset[150::2, 150::2]
    print("Sparse slice from dataset:", slice_, sep='\n')
    print("Sparse slice from input array:", data[150::2, 150::2], sep='\n')

# Disabling Blosc2 optimized slicing
# ----------------------------------
# Just set the `BLOSC2_FILTER` environment variable to some non-zero integer.
print("Disabling Blosc2 optimized slicing via the environment.")
os.environ['BLOSC2_FILTER'] = '1'
with h5py.File(file_name, 'r') as f:
    # If support for Blosc2 optimized slicing is available,
    # there is no need to import anything else explicitly for reading.
    # However, if Python-Blosc2 is not available on your system,
    # you need to import hdf5plugin to access Blosc2-compressed data
    # (without optimizations).
    import hdf5plugin
    dataset = f[dataset_name]
    slice_ = dataset[150:, 150:]
    print("Slice from dataset:", slice_, sep='\n')
    print("Slice from input array:", data[150:, 150:], sep='\n')
