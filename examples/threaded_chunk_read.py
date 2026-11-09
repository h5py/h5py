"""Read a compressed dataset with several threads, bypassing HDF5's filters

HDF5 decompresses chunks while h5py holds its global lock, so reading datasets
from several threads does not make decompression any faster. zlib releases the
GIL though. So read the raw chunks one after the other, then decompress them in
a thread pool. This only pays off if decompression takes most of the read time.
"""
from concurrent.futures import ThreadPoolExecutor
import os
from time import perf_counter
import zlib

import h5py
import numpy as np


def read_gzip_threaded(dataset, max_workers=None):
    """Same result as dataset[()] for a dataset compressed with gzip only"""
    filters = dataset.id.get_create_plist().get_nfilters()
    if dataset.compression != "gzip" or filters != 1:
        raise ValueError("Only for datasets with the gzip filter alone")
    if dataset.dtype.hasobject or dataset.dtype.subdtype:
        raise ValueError("Only for datasets with a plain fixed size element type")
    if dataset.size == 0:
        return dataset[()]
    slices = list(dataset.iter_chunks())
    if dataset.id.get_num_chunks() != len(slices):
        # read_direct_chunk() fails on chunks that were never written
        raise ValueError("Only for datasets with all chunks written")

    raw_chunks = [
        dataset.id.read_direct_chunk(tuple(s.start for s in chunk_slice))
        for chunk_slice in slices
    ]
    result = np.empty(dataset.shape, dtype=dataset.dtype)

    def decompress(chunk_slice, raw_chunk):
        filter_mask, stored = raw_chunk
        # A set bit means the filter was skipped for this chunk, for example
        # if it was written with write_direct_chunk(filter_mask=1).
        data = stored if filter_mask & 1 else zlib.decompress(stored)
        chunk = np.frombuffer(data, dtype=dataset.dtype).reshape(dataset.chunks)
        # Chunks at the end of the dataset are stored with their full size
        inside = tuple(slice(s.stop - s.start) for s in chunk_slice)
        result[chunk_slice] = chunk[inside]

    with ThreadPoolExecutor(max_workers) as pool:
        list(pool.map(decompress, slices, raw_chunks))
    return result


if __name__ == "__main__":
    # A signal with noise compresses like many real measurements
    values = np.sin(np.arange(40_000_000) / 1000.0)
    values += np.random.default_rng(0).normal(scale=1e-3, size=values.size)

    with h5py.File("threaded_chunk_read.h5", "w") as f:
        f.create_dataset("data", data=values, chunks=(2**18,), compression="gzip")

    with h5py.File("threaded_chunk_read.h5", "r") as f:
        start = perf_counter()
        expected = f["data"][()]
        print(f"h5py:    {perf_counter() - start:.2f} s")

        start = perf_counter()
        threaded = read_gzip_threaded(f["data"])
        print(f"threads: {perf_counter() - start:.2f} s on {os.cpu_count()} CPUs")

    np.testing.assert_array_equal(threaded, expected)
    os.remove("threaded_chunk_read.h5")
