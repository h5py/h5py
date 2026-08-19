# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Tests for the versioning helpers in h5py.version
"""

import pytest

from h5py.version import _hdf5_abi_compatible


@pytest.mark.parametrize('built, running', [
    # Identical versions are always fine
    ((1, 14, 6), (1, 14, 6)),
    ((2, 0, 0), (2, 0, 0)),
    # HDF5 promises ABI compatibility with later releases of the same major
    # version from 2.0 onwards
    ((2, 0, 0), (2, 0, 1)),
    ((2, 1, 0), (2, 2, 0)),
    ((2, 0, 0), (2, 11, 3)),
])
def test_abi_compatible(built, running):
    assert _hdf5_abi_compatible(built, running)


@pytest.mark.parametrize('built, running', [
    # Older libraries may lack symbols h5py was built against
    ((2, 2, 0), (2, 1, 0)),
    ((2, 0, 1), (2, 0, 0)),
    # Nothing is promised across a major version
    ((2, 0, 0), (3, 0, 0)),
    ((2, 0, 0), (1, 14, 6)),
    ((1, 14, 6), (2, 0, 0)),
    # 1.x predates the ABI compatibility promise
    ((1, 14, 3), (1, 14, 6)),
    ((1, 14, 6), (1, 14, 3)),
    ((1, 12, 3), (1, 14, 3)),
])
def test_abi_incompatible(built, running):
    assert not _hdf5_abi_compatible(built, running)
