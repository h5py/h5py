'''Concatenate multiple files into a single virtual dataset
'''
import h5py
import numpy as np
import sys
import os


def concatenate(file_names_to_concatenate):
    entry_key = 'data'  # where the data is inside of the source files.
    sh = h5py.File(file_names_to_concatenate[0], 'r')[entry_key].shape  # get the first ones shape.
    layout = h5py.VirtualLayout(shape=(len(file_names_to_concatenate),) + sh,
                                dtype=np.float64)
    with h5py.File("VDS.h5", 'w', libver='latest') as f:
        for i, filename in enumerate(file_names_to_concatenate):
            vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
            layout[i, :, :, :] = vsource

        f.create_virtual_dataset(entry_key, layout, fillvalue=0)


def create_random_file(folder, index):
    """create one random file"""
    name = os.path.join(folder, 'myfile_' + str(index))
    with h5py.File(name=name, mode='w') as f:
        d = f.create_dataset('data', (5, 10, 20), 'i4')
        data = np.random.randint(low=0, high=100, size=(5*10*20))
        data = data.reshape(5, 10, 20)
        d[:] = data
    return name


def main(argv):
    files = argv[1:]
    if len(files) == 0:
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        for i_file in range(5):
            files.append(create_random_file(tmp_dir, index=i_file))
    concatenate(files)


if __name__ == '__main__':
    main(sys.argv)
