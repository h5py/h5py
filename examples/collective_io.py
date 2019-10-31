# This file is to test collective io in h5py

"""
Author:  Jialin Liu, jalnliu@lbl.gov
Date:    Nov 17, 2015
Prerequisites: python 2.5.0, mpi4py and numpy
Source Codes: Already submit this 'collective io' branch to h5py master, meanwhile, can download this branch at https://github.com/valiantljk/h5py.git
Note: Must build the h5py with parallel hdf5
"""

from mpi4py import MPI
import numpy as np
import h5py
import sys

#"run as "mpirun -np 64 python-mpi collective_io.py 1 file.h5"
#(1 is for collective write, other numbers for non-collective write)"

colw=1 #default is collective write
filename="parallel_test.hdf5"
if len(sys.argv)>2:
    colw = int(sys.argv[1])
    filename=str(sys.argv[2])
comm =MPI.COMM_WORLD
nproc = comm.Get_size()
f = h5py.File(filename, 'w', driver='mpio', comm=MPI.COMM_WORLD)
rank = comm.Get_rank()
length_x = 6400*1024
length_y = 1024
dset = f.create_dataset('test', (length_x,length_y), dtype='f8')
#data type should be consistent in numpy and h5py, e.g., 64 bits
#otherwise, hdf5 layer will fall back to independent io.
f.atomic = False
length_rank=length_x / nproc
length_last_rank=length_x -length_rank*(nproc-1)
comm.Barrier()
timestart=MPI.Wtime()
start=rank*length_rank
end=start+length_rank
if rank==nproc-1: #last rank
    end=start+length_last_rank
temp=np.random.random((end-start,length_y))
if colw==1:
    with dset.collective:
        dset[start:end,:] = temp
else:
    dset[start:end,:] = temp
comm.Barrier()
timeend=MPI.Wtime()
if rank==0:
    if colw==1:
        print("collective write time %f" %(timeend-timestart))
    else:
        print("independent write time %f" %(timeend-timestart))
    print("data size x: %d y: %d" %(length_x, length_y))
    print("file size ~%d GB" % (length_x*length_y/1024.0/1024.0/1024.0*8.0))
    print("number of processes %d" %nproc)
f.close()
