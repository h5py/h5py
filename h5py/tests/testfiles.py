#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

"""
    Contains code to "bootstrap" HDF5 files for the test suite.  Since it
    uses the same code that will eventually be tested, the files it produces
    are manually inspected using HDFView, and distributed with the package.
"""

import numpy as np
import h5py

class Group(dict):

    def __init__(self, *args, **kwds):
        dict.__init__(self, *args, **kwds)
        self.attrs = {}

class File(Group):

    def __init__(self, name, *args, **kwds):
        Group.__init__(self, *args, **kwds)
        self.name = name

class Dataset(object):

    argnames = ('shape', 'dtype', 'data','chunks', 'compression', 'shuffle',
                'fletcher32', 'maxshape')

    def __init__(self, *args, **kwds):

        kwds.update(zip(self.argnames, args))
        self.kwds = kwds
        self.attrs = {}

class Datatype(object):
     
    def __init__(self, dtype):
        self.dtype = dtype
        self.attrs = {}

def compile_hdf5(fileobj):
    """ Take a "model" HDF5 tree and write it to an actual file. """

    def update_attrs(hdf_obj, attrs_dict):
        for name in sorted(attrs_dict):
            val = attrs_dict[name]
            hdf_obj.attrs[name] = val

    def store_dataset(group, name, obj):
        """ Create and store a dataset in the given group """
        dset = group.create_dataset(name, **obj.kwds)
        update_attrs(dset, obj.attrs)

    def store_type(group, name, obj):
        """ Commit the given datatype to the group """
        group[name] = obj.dtype
        htype = group[name]
        update_attrs(htype, obj.attrs)

    def store_group(group, name, obj):
        """ Create a new group inside this existing group. """

        # First create the new group (if it's not the root group)
        if name is not None:
            hgroup = group.create_group(name)
        else:
            hgroup = group

        # Now populate it
        for new_name in sorted(obj):
            new_obj = obj[new_name]

            if isinstance(new_obj, Dataset):
                store_dataset(hgroup, new_name, new_obj)
            elif isinstance(new_obj, Datatype):
                store_type(hgroup, new_name, new_obj)
            elif isinstance(new_obj, Group):
                store_group(hgroup, new_name, new_obj)

        update_attrs(hgroup, obj.attrs)

    f = h5py.File(fileobj.name, 'w')
    store_group(f['/'], None, fileobj)
    f.close()


def file_attrs():
    """ "Attributes" test file (also used by group tests) """
    sg1 = Group()
    sg2 = Group()
    sg3 = Group()
    grp = Group({'Subgroup1': sg1, 'Subgroup2': sg2, 'Subgroup3': sg3})
    grp.attrs =  {'String Attribute': np.asarray("This is a string.", '|S18'),
                  'Integer': np.asarray(42, '<i4'),
                  'Integer Array': np.asarray([0,1,2,3], '<i4'),
                  'Byte': np.asarray(-34, '|i1') }

    return File('attributes.hdf5', {'Group': grp})

def file_dset():
    """ "Dataset" test file.  Bears a suspicious resemblance to a certain
        PyTables file.
    """
    dtype = np.dtype(
        [('a_name','>i4'),
         ('c_name','|S6'),
         ('d_name', np.dtype( ('>i2', (5,10)) )),
         ('e_name', '>f4'),
         ('f_name', np.dtype( ('>f8', (10,)) )),
         ('g_name', '<u1')])

    arr = np.ndarray((6,), dtype)
    for i in range(6):
        arr[i]["a_name"] = i,
        arr[i]["c_name"] = "Hello!"
        arr[i]["d_name"][:] = np.sum(np.indices((5,10)),0) + i
        arr[i]["e_name"] = 0.96*i
        arr[i]["f_name"][:] = np.array((1024.9637*i,)*10)
        arr[i]["g_name"] = 109

    dset = Dataset(data=arr, chunks=(3,))

    return File('smpl_compound_chunked.hdf5', {'CompoundChunked': dset})

if __name__ == '__main__':
    compile_hdf5(file_attrs())
    compile_hdf5(file_dset())








