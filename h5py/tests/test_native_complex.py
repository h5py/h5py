# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
Testing native complex number datatypes.
"""

import sys
import numpy as np
import pytest
import h5py
from h5py.h5 import get_config  # type: ignore
from h5py import h5t
from .data_files import get_data_file_path

cfg = get_config()

pytestmark = [
    pytest.mark.skipif(
        h5py.version.hdf5_version_tuple < (2, 0, 0),
        reason="Requires HDF5 >= 2.0",
    ),
    pytest.mark.skipif(
        not cfg.native_complex,
        reason="Native HDF5 complex number datatypes not available",
    ),
]


def test_hdf5_dtype():
    """Low-level checks of HDF5 native complex number datatypes"""
    sys_bo = h5t.ORDER_BE if sys.byteorder == "big" else h5t.ORDER_LE
    cases = [
        (h5t.COMPLEX_IEEE_F16LE, 4, h5t.ORDER_LE),
        (h5t.COMPLEX_IEEE_F16BE, 4, h5t.ORDER_BE),
        (h5t.COMPLEX_IEEE_F32LE, 8, h5t.ORDER_LE),
        (h5t.COMPLEX_IEEE_F32BE, 8, h5t.ORDER_BE),
        (h5t.COMPLEX_IEEE_F64LE, 16, h5t.ORDER_LE),
        (h5t.COMPLEX_IEEE_F64BE, 16, h5t.ORDER_BE),
        (h5t.NATIVE_FLOAT_COMPLEX, 8, sys_bo),
        (h5t.NATIVE_DOUBLE_COMPLEX, 16, sys_bo),
    ]
    if hasattr(np, "complex256"):
        cases.append((h5t.NATIVE_LDOUBLE_COMPLEX, 32, sys_bo))
    else:
        cases.append((h5t.NATIVE_LDOUBLE_COMPLEX, 16, sys_bo))
    for h5type, size, order in cases:
        assert isinstance(h5type, h5t.TypeComplexID)
        assert h5type.get_size() == size
        assert h5type.get_order() == order


def test_cmplx_type_trnslt():
    """Translate native HDF5 complex number datatype to and from NumPy dtype"""
    cases = [
        (h5t.COMPLEX_IEEE_F32LE, np.dtype("<c8")),
        (h5t.COMPLEX_IEEE_F32BE, np.dtype(">c8")),
        (h5t.COMPLEX_IEEE_F64LE, np.dtype("<c16")),
        (h5t.COMPLEX_IEEE_F64BE, np.dtype(">c16")),
        (h5t.NATIVE_FLOAT_COMPLEX, np.dtype("=c8")),
        (h5t.NATIVE_DOUBLE_COMPLEX, np.dtype("=c16")),
    ]
    if hasattr(np, "complex256"):
        cases.append((h5t.NATIVE_LDOUBLE_COMPLEX, np.dtype("=c32")))
    for h5type, dt in cases:
        assert h5type.dtype == dt
        assert h5t.py_create(dt) == h5type


def test_create_dset_and_attr(writable_file):
    """Create complex number datasets and attributes of different datatypes"""
    for dt in ("<c8", ">c8", "<c16", ">c16"):
        complex_array = (np.random.rand(100) + 1j * np.random.rand(100)).astype(dt)
        ds = writable_file.create_dataset(dt, data=complex_array)
        c = np.array(1.9 + 1j * 6.7, dtype=dt)
        ds.attrs["c"] = c
        assert isinstance(ds.id.get_type(), h5t.TypeComplexID)
        assert isinstance(ds.attrs.get_id("c").get_type(), h5t.TypeComplexID)
        np.testing.assert_array_equal(ds[...], complex_array)
        np.testing.assert_array_equal(ds.attrs["c"], c)


def test_dtype_cmpnd_cmplx():
    """Check the resulting dtype of the two-field compound data as complex numbers"""
    with h5py.File(get_data_file_path("compound-dtype-complex.h5"), mode="r") as f:
        for obj in f.values():
            if isinstance(obj, h5py.Dataset):
                stored_dtype = obj.id.get_type()
                np_dtype = obj.dtype
                assert isinstance(stored_dtype, h5t.TypeCompoundID)
                assert stored_dtype.get_nmembers() == 2
                assert stored_dtype.get_member_name(0) == cfg._r_name
                assert stored_dtype.get_member_name(1) == cfg._i_name
                assert isinstance(stored_dtype.get_member_type(0), h5t.TypeFloatID)
                assert isinstance(stored_dtype.get_member_type(1), h5t.TypeFloatID)
                assert np_dtype.kind == "c"
                assert (
                    stored_dtype.get_member_type(0).get_size()
                    + stored_dtype.get_member_type(1).get_size()
                ) == np_dtype.itemsize

                if np_dtype.byteorder == ">" or (
                    sys.byteorder == "big" and np_dtype.byteorder == "="
                ):
                    assert stored_dtype.get_member_type(0).get_order() == h5t.ORDER_BE
                    assert stored_dtype.get_member_type(1).get_order() == h5t.ORDER_BE
                elif np_dtype.byteorder == "<" or (
                    sys.byteorder == "little" and np_dtype.byteorder == "="
                ):
                    assert stored_dtype.get_member_type(0).get_order() == h5t.ORDER_LE
                    assert stored_dtype.get_member_type(1).get_order() == h5t.ORDER_LE
