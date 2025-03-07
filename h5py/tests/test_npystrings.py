import numpy as np
import pytest

import h5py

NUMPY_GE2 = int(np.__version__.split(".")[0]) >= 2
pytestmark = pytest.mark.skipif(not NUMPY_GE2, reason="requires numpy >=2.0")


def test_roundtrip(writable_file):
    ds = writable_file.create_dataset("x", shape=(2, 2), dtype="T")
    # The h5py.Dataset object remembers its dtype
    assert ds.dtype.kind == "T"
    data = [["foo", "bar"], ["hello world", ""]]
    ds[:] = data
    a = ds[:]
    assert a.dtype.kind == "T"
    np.testing.assert_array_equal(a, data)

    # Recreating the h5py.Dataset object resets the dtype to the default;
    # because it's not stored in the HDF5 file, it's not recoverable.
    ds = writable_file["x"]
    assert ds.dtype == object
    np.testing.assert_array_equal(ds.asstr()[:], data)

    ds = ds.astype("T")
    assert isinstance(ds, h5py.Dataset)  # Not an AsTypeView
    ds[0, 0] = "baz"  # Unlike an AsTypeView, it's writeable
    data[0][0] = "baz"
    a = ds[:]
    assert a.dtype.kind == "T"
    np.testing.assert_array_equal(a, data)

    ds[0, 0] = np.asarray("123", dtype="O")
    data[0][0] = "123"
    np.testing.assert_array_equal(ds[:], data)


def test_fromdata(writable_file):
    data = [["foo", "bar"]]
    np_data = np.asarray(data, dtype="T")
    x = writable_file.create_dataset("x", data=data, dtype="T")
    y = writable_file.create_dataset("y", data=data, dtype=np.dtypes.StringDType())
    z = writable_file.create_dataset("z", data=np_data)

    for ds in (x, y, z):
        assert ds.dtype.kind == "T"
        np.testing.assert_array_equal(ds[:], np_data)
    for name in ("x", "y", "z"):
        ds = writable_file[name]
        assert ds.dtype == object
        np.testing.assert_array_equal(ds.asstr()[:], data)
        ds = ds.astype("T")
        assert ds.dtype.kind == "T"
        a = ds[:]
        assert a.dtype.kind == "T"
        np.testing.assert_array_equal(a, data)


def test_astype_is_reversible(writable_file):
    data = ["foo", "bar"]
    x = writable_file.create_dataset(
        "x", data=data, dtype=h5py.string_dtype()
    )
    assert x.dtype == object
    x = x.astype("T")
    assert x.dtype.kind == "T"
    assert x[:].dtype.kind == "T"

    y = x.astype(object)
    z = x.astype("O")
    assert y.dtype == object
    assert z.dtype == object
    assert y[:].dtype == object
    assert z[:].dtype == object

    # asstr() internally calls astype
    w = x.asstr()
    assert w.dtype == object
    np.testing.assert_array_equal(w[:], data)


def test_fixed_to_variable_width(writable_file):
    data = ["foo", "longer than 8 bytes"]
    x = writable_file.create_dataset(
        "x", data=data, dtype=h5py.string_dtype(length=20)
    )
    assert x.dtype == "S20"

    # read T <- S
    y = x.astype("T")
    assert isinstance(y, h5py.Dataset)
    assert y.dtype.kind == "T"
    assert y[:].dtype.kind == "T"
    np.testing.assert_array_equal(y[:], data)

    # write T -> S
    x[0] = np.asarray("1234", dtype="T")
    data[0] = "1234"
    np.testing.assert_array_equal(y[:], data)


def test_fixed_to_variable_width_too_short(writable_file):
    data = ["foo", "bar"]
    x = writable_file.create_dataset(
        "x", data=data, dtype=h5py.string_dtype(length=3)
    )
    assert x.dtype == "S3"

    # write T -> S
    x[0] = np.asarray("1234", dtype="T")
    np.testing.assert_array_equal(x[:], [b"123", b"bar"])


def test_variable_to_fixed_width(writable_file):
    data = ["foo", "longer than 8 bytes"]
    bdata = [b"foo", b"longer than 8 bytes"]
    x = writable_file.create_dataset("x", data=data, dtype="T")
    assert x.dtype.kind == "T"

    # read S <- T
    y = x.astype("S20")
    assert y.dtype == "S20"
    assert y[:].dtype == "S20"
    np.testing.assert_array_equal(y[:], bdata)

    y = x.astype("S3")
    assert y.dtype == "S3"
    assert y[:].dtype == "S3"
    np.testing.assert_array_equal(y[:], [b"foo", b"lon"])

    # write S -> T
    x[0] = np.asarray(b"1234", dtype="S5")
    data[0] = "1234"
    np.testing.assert_array_equal(x[:], data)


def test_write_object_into_npystrings(writable_file):
    x = writable_file.create_dataset("x", data=["foo"], dtype="T")
    assert x.dtype.kind == "T"
    x[0] = np.asarray("1234", dtype="O")
    np.testing.assert_array_equal(x[:], "1234")


def test_write_npystrings_into_object(writable_file):
    x = writable_file.create_dataset(
        "x", data=["foo"], dtype=h5py.string_dtype()
    )
    assert x.dtype == object
    x[0] = np.asarray("1234", dtype="T")
    np.testing.assert_array_equal(x[:], b"1234")


def test_repr(writable_file):
    x = writable_file.create_dataset("x", data=["foo"])
    assert repr(x) == '<HDF5 dataset "x": shape (1,), type "|O">'
    x = x.astype("T")
    assert repr(x) == '<HDF5 dataset "x": shape (1,), type StringDType()>'
