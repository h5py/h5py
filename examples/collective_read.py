# This file is to test collective io in h5py
 
"""
Author:  Jialin Liu, jalnliu@lbl.gov
Date:    Nov 17, 2015
Prerequisites: python 2.5.0, mpi4py and numpy
Source Codes: This 'collective io' branch is pushed into the h5py master
Note: Must build the h5py with parallel hdf5
"""

from mpi4py import MPI
import numpy as np
import h5py
import time
import sys

#"run as "mpirun -np 64 python-mpi collective_io.py 1 file.h5" 
#(1 is for collective write, ohter number for non-collective write)"

filename="parallel_test.hdf5"
if len(sys.argv)>2:
	filename=str(sys.argv[1])
	dataset =str(sys.argv[2])
        colr = int(sys.argv[3])
comm =MPI.COMM_WORLD
print 'colr:%d'%colr
nproc = comm.Get_size()
f = h5py.File(filename, 'r', driver='mpio', comm=MPI.COMM_WORLD)
rank = comm.Get_rank()
#open the dataset, get the handle
dset = f[dataset]

length_x = dset.shape[0]
length_y = dset.shape[1]

print (length_x,length_y)
f.atomic = False
length_rank=length_x / nproc
length_last_rank=length_x -length_rank*(nproc-1)
comm.Barrier()
timestart=MPI.Wtime()
# start and end for each rank, 
start=rank*length_rank
end=start+length_rank
if rank==nproc-1: #adjust last rank
    end=start+length_last_rank

#creat an empty numpy array for storing the data
temp =np.empty(dset.shape,dset.dtype)

# Do the independent I/O
if colr==0: 
 temp[start:end,:] = dset[start:end,:]
else:
# Do the collective I/O
 with dset.collective:
   temp[start:end,:] = dset[start:end,:]
comm.Barrier()
print "rank: ",rank,"\n", temp[start:end,:]
timeend=MPI.Wtime()
if rank==0:
    if colr==0:
     print "independent read time %f" %(timeend-timestart)
    else:
     print "collective read time %f" %(timeend-timestart)
    print "data size x: %d y: %d" %(length_x, length_y)
    print "file size ~%d GB" % (length_x*length_y/1024.0/1024.0/1024.0*8.0)
    print "number of processes %d" %nproc
f.close()
