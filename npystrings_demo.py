# DNM
import h5py

f = h5py.File("foo.h5", "w")
data = ["foo", "Hello world this is a very long string indeed", "bar"]
f.create_dataset("data", data=data, dtype=h5py.string_dtype())
ds = f["data"]
a = ds[:]
print(repr(a))
