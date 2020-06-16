import h5py
import numpy as np

arr = np.array([np.datetime64('2019-09-22T17:38:30')])

with h5py.File('datetimes.h5', 'w') as f:
    # Create dataset
    f['data'] = arr.astype(h5py.opaque_dtype(arr.dtype))

    # Read
    print(f['data'][:])
