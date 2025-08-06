import numpy as np
import pytest

import h5py

NUMPY_GE2 = int(np.__version__.split(".")[0]) >= 2
pytestmark = pytest.mark.skipif(not NUMPY_GE2, reason="requires numpy >=2.0")


def test_create_with_dtype_T(writable_file):
    ds = writable_file.create_dataset("x", shape=(2, 2), dtype="T")
    data = [["foo", "bar"], ["hello world", ""]]
    ds[:] = data
    a = ds.asstr()[:]
    np.testing.assert_array_equal(a, data)

    ds = writable_file["x"]
    assert ds.dtype == object
    np.testing.assert_array_equal(ds.asstr()[:], data)

    ds[0, 0] = "baz"
    data[0][0] = "baz"
    a = ds.astype("T")[:]
    assert a.dtype.kind == "T"
    np.testing.assert_array_equal(a, data)

    ds[0, 0] = np.asarray("123", dtype="O")
    data[0][0] = "123"
    np.testing.assert_array_equal(ds.asstr()[:], data)


def test_fromdata(writable_file):
    data = [["foo", "bar"]]
    np_data = np.asarray(data, dtype="T")
    x = writable_file.create_dataset("x", data=data, dtype="T")
    y = writable_file.create_dataset("y", data=data, dtype=np.dtypes.StringDType())
    z = writable_file.create_dataset("z", data=np_data)

    for ds in (x, y, z):
        assert ds.dtype.kind == "O"
        np.testing.assert_array_equal(ds.astype("T")[:], np_data)
    for name in ("x", "y", "z"):
        ds = writable_file[name]
        assert ds.dtype == object
        np.testing.assert_array_equal(ds.asstr()[:], data)
        ds = ds.astype("T")
        assert ds.dtype.kind == "T"
        a = ds[:]
        assert a.dtype.kind == "T"
        np.testing.assert_array_equal(a, data)


def test_fixed_to_variable_width(writable_file):
    data = ["foo", "longer than 8 bytes"]
    x = writable_file.create_dataset(
        "x", data=data, dtype=h5py.string_dtype(length=20)
    )
    assert x.dtype == "S20"

    # read T <- S
    y = x.astype("T")
    assert y.dtype.kind == "T"
    assert y[:].dtype.kind == "T"
    np.testing.assert_array_equal(y[:], data)

    # write T -> S
    x[0] = np.asarray("1234", dtype="T")
    data[0] = "1234"
    np.testing.assert_array_equal(y[:], data)


def test_fixed_to_variable_width_too_short(writable_file):
    # Note: this test triggers calls to H5Tconvert which are otherwise skipped.

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
    bdata[0] = b"1234"
    np.testing.assert_array_equal(x[:], bdata)


def test_write_object_into_npystrings(writable_file):
    x = writable_file.create_dataset("x", data=["foo"], dtype="T")
    x[0] = np.asarray("1234", dtype="O")
    np.testing.assert_array_equal(x[:], b"1234")


def test_write_npystrings_into_object(writable_file):
    x = writable_file.create_dataset(
        "x", data=["foo"], dtype=h5py.string_dtype()
    )
    assert x.dtype == object
    x[0] = np.asarray("1234", dtype="T")
    np.testing.assert_array_equal(x[:], b"1234")

    # Test with HDF5 variable-length strings with ASCII character set
    xa = writable_file.create_dataset(
        "xa", shape=(1,), dtype=h5py.string_dtype('ascii')
    )
    xa[0] = np.asarray("2345", dtype="T")
    np.testing.assert_array_equal(xa[:], b"2345")


def test_fillvalue(writable_file):
    # Create as NpyString dtype
    x = writable_file.create_dataset("x", shape=(2,), dtype="T", fillvalue="foo")
    assert isinstance(x.fillvalue, bytes)
    assert x.fillvalue == b"foo"
    assert x[0] == b"foo"

    # Create as object dtype
    y = writable_file.create_dataset(
        "y", shape=(2,), dtype=h5py.string_dtype(), fillvalue=b"foo"
    )
    assert isinstance(y.fillvalue, bytes)
    assert y.fillvalue == b"foo"
    assert y[0] == b"foo"
    # Convert object dtype to NpyString
    y = y.astype("T")
    assert y[0] == "foo"


def test_empty_string(writable_file):
    data = np.array(["", "a", "b"], dtype="T")
    x = writable_file.create_dataset("x", data=data)
    np.testing.assert_array_equal(x[:], [b"", b"a", b"b"])
    np.testing.assert_array_equal(x.astype("T")[:], data)
    data[:2] = ["c", ""]
    x[:2] = data[:2]
    np.testing.assert_array_equal(x[:], [b"c", b"", b"b"])
    np.testing.assert_array_equal(x.astype("T")[:], data)


def test_astype_nonstring(writable_file):
    x = writable_file.create_dataset("x", shape=(2, ), dtype="i8")
    with pytest.raises(TypeError, match="HDF5 string datatype"):
        x.astype("T")


def test_resized_read(writable_file):
    """Read default values created by resize(). This triggers a special case
    where libhdf5 returns a char** containing NULL pointers.
    """
    l = ["string1", "string2", "string3"]
    data = np.array(l, dtype='T')
    d = writable_file.create_dataset("dset", data=data, maxshape=(None,))
    d.resize((10,))

    np.testing.assert_array_equal(d[:], np.array(
        [s.encode() for s in l] + [b''] * 7, dtype=object
    ))
    np.testing.assert_array_equal(d.astype('T')[:], np.array(l + [''] * 7, dtype='T'))
