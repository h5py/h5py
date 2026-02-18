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
        not cfg.has_native_complex,
        reason="Native HDF5 complex number datatypes not available",
    ),
]

REQUIRE_NUMPY_COMPLEX256 = pytest.mark.skipif(
    not hasattr(np, "complex256"),
    reason="complex256 type is not available in numpy",
)

NATIVE_BYTE_ORDER = h5t.ORDER_BE if sys.byteorder == "big" else h5t.ORDER_LE


@pytest.mark.parametrize(
    "h5type_str, size, order",
    [
        pytest.param("COMPLEX_IEEE_F16LE", 4, h5t.ORDER_LE),
        pytest.param("COMPLEX_IEEE_F16BE", 4, h5t.ORDER_BE, id="F16BE"),
        pytest.param("COMPLEX_IEEE_F32LE", 8, h5t.ORDER_LE, id="F32LE"),
        pytest.param("COMPLEX_IEEE_F32BE", 8, h5t.ORDER_BE, id="F32BE"),
        pytest.param("COMPLEX_IEEE_F64LE", 16, h5t.ORDER_LE, id="F64LE"),
        pytest.param("COMPLEX_IEEE_F64BE", 16, h5t.ORDER_BE, id="F64BE"),
        pytest.param("NATIVE_FLOAT_COMPLEX", 8, NATIVE_BYTE_ORDER, id="float-native"),
        pytest.param("NATIVE_DOUBLE_COMPLEX", 16, NATIVE_BYTE_ORDER, id="double-native"),
        pytest.param(
            "NATIVE_LDOUBLE_COMPLEX", 32, NATIVE_BYTE_ORDER,
            id="long-native",
            marks=REQUIRE_NUMPY_COMPLEX256,
        )
    ]
)
def test_hdf5_dtype(h5type_str, size, order):
    """Low-level checks of HDF5 native complex number datatypes"""
    h5type = getattr(h5t, h5type_str)
    assert isinstance(h5type, h5t.TypeComplexID)
    assert h5type.get_size() == size
    assert h5type.get_order() == order

H5_VS_NUMPY_DTYPES = [
    pytest.param("COMPLEX_IEEE_F32LE", "<c8", id="<c8"),
    pytest.param("COMPLEX_IEEE_F32BE", ">c8", id=">c8"),
    pytest.param("COMPLEX_IEEE_F64LE", "<c16", id="<c16"),
    pytest.param("COMPLEX_IEEE_F64BE", ">c16", id=">c16"),
    pytest.param("NATIVE_FLOAT_COMPLEX", "=c8", id="=c8"),
    pytest.param("NATIVE_DOUBLE_COMPLEX", "=c16", id="=c16"),
    pytest.param(
        "NATIVE_LDOUBLE_COMPLEX", "=c32", id="=c32",
        marks=REQUIRE_NUMPY_COMPLEX256,
    ),
]

@pytest.mark.parametrize("h5type_str, dt", H5_VS_NUMPY_DTYPES)
def test_cmplx_type_trnslt(h5type_str, dt):
    """Translate native HDF5 complex number datatype to NumPy dtype"""
    assert getattr(h5t, h5type_str).dtype == np.dtype(dt)


def check_compound_complex_datatype(datatype, np_dtype):
    """Check h5py's old complex number type"""
    assert isinstance(datatype, h5t.TypeCompoundID)
    assert datatype.get_nmembers() == 2
    assert datatype.get_member_name(0) == cfg._r_name
    assert datatype.get_member_name(1) == cfg._i_name
    assert isinstance(datatype.get_member_type(0), h5t.TypeFloatID)
    assert isinstance(datatype.get_member_type(1), h5t.TypeFloatID)

    assert (
       datatype.get_member_type(0).get_size()
       + datatype.get_member_type(1).get_size()
    ) == np_dtype.itemsize

    match (np_dtype.byteorder, sys.byteorder):
        case (">", _) | ("=", "big"):
            expected = h5t.ORDER_BE
        case ("<", _) | ("=", "little"):
            expected = h5t.ORDER_LE
        case _ as _unreachable:
            raise AssertionError

    assert datatype.get_member_type(0).get_order() == expected
    assert datatype.get_member_type(1).get_order() == expected


@pytest.mark.parametrize("dt", ["<c8", ">c8", "<c16", ">c16"])
def test_default_create(writable_file, dt):
    """Test default translation when creating datasets & attributes"""
    complex_array = (np.random.rand(100) + 1j * np.random.rand(100)).astype(dt)
    ds = writable_file.create_dataset(dt, data=complex_array)
    c = np.array(1.9 + 1j * 6.7, dtype=dt)
    ds.attrs["c"] = c
    check_compound_complex_datatype(ds.id.get_type(), np.dtype(dt))
    check_compound_complex_datatype(ds.attrs.get_id("c").get_type(), np.dtype(dt))
    np.testing.assert_array_equal(ds[...], complex_array)
    np.testing.assert_array_equal(ds.attrs["c"], c)


@pytest.mark.parametrize("h5type_str, dt", H5_VS_NUMPY_DTYPES)
def test_explicit_create(writable_file, h5type_str, dt):
    """Explicitly use native complex datatype to create datasets & attributes"""
    h5type = getattr(h5t, h5type_str)
    np_dt = np.dtype(dt)
    complex_array = (np.random.rand(100) + 1j * np.random.rand(100)).astype(np_dt)
    ds = writable_file.create_dataset(dt, (100,), dtype=h5py.Datatype(h5type))
    ds[:] = complex_array
    c = np.array(1.9 + 1j * 6.7, dtype=np_dt)
    ds.attrs.create("c", c, dtype=h5py.Datatype(h5type))
    assert isinstance(ds.id.get_type(), h5t.TypeComplexID)
    assert isinstance(ds.attrs.get_id("c").get_type(), h5t.TypeComplexID)
    np.testing.assert_array_equal(ds[...], complex_array)
    np.testing.assert_array_equal(ds.attrs["c"], c)


def test_dtype_cmpnd_cmplx():
    """Check the resulting dtype of the two-field compound data as complex numbers"""
    with h5py.File(get_data_file_path("compound-dtype-complex.h5"), mode="r") as f:
        for obj in filter(lambda obj: isinstance(obj, h5py.Dataset), f.values()):
            check_compound_complex_datatype(obj.id.get_type(), obj.dtype)
