import h5py 
f=h5py.File('test.h5','w')
import numpy as np
arr = np.arange(100)
dset=f.create_dataset('dset',(10,10),data=arr)
f.close()

