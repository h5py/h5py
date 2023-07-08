"""
An example of using MultiBlockSlice to interleave frames in a virtual dataset.
"""
import h5py


# These files were written round-robin in blocks of 1000 frames from a single source
# 1.h5 has 0-999, 4000-4999, ...; 2.h4 has 1000-1999, 5000-5999, ...; etc
# The frames from each block are contiguous within each file
# e.g. 1.h5 = [..., 998, 999, 4000, 4001, ... ]
files = ["1.h5", "2.h5", "3.h5", "4.h5"]
dataset_name = "data"
dtype = "float"
source_shape = (25000, 256, 512)
target_shape = (100000, 256, 512)
block_size = 1000

v_layout = h5py.VirtualLayout(shape=target_shape, dtype=dtype)

for file_idx, file_path in enumerate(files):
    v_source = h5py.VirtualSource(
        file_path, name=dataset_name, shape=source_shape, dtype=dtype
    )
    dataset_frames = v_source.shape[0]

    # A single raw file maps to every len(files)th block of frames in the VDS
    start = file_idx * block_size  # 0, 1000, 2000, 3000
    stride = len(files) * block_size  # 4000
    count = dataset_frames // block_size  # 25
    block = block_size  # 1000

    # MultiBlockSlice for frame dimension and full extent for height and width
    v_layout[h5py.MultiBlockSlice(start, stride, count, block), :, :] = v_source

with h5py.File("interleave_vds.h5", "w", libver="latest") as f:
    f.create_virtual_dataset(dataset_name, v_layout, fillvalue=0)
