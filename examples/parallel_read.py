# This file is to test collective io in h5py
 
"""
Author:  Jialin Liu, jalnliu@lbl.gov
Date:    July 11, 2017
Prerequisites: python 2.7.0, mpi4py and numpy
This code does parallel read (a 2D array) with independent I/O or collective IO
Parameters: filename datasetname is_collective
Example: mpirun -np 10 python-mpi parallel_read.py test.h5 dset 0 
0 for independent IO
1 for collective IO
"""

from mpi4py import MPI
import numpy as np
import h5py
import time
import sys

if len(sys.argv)>3:
   filename=str(sys.argv[1])
   dataset =str(sys.argv[2])
   colr = int(sys.argv[3])
else:
   print "args: filename dataset is_collective?"
   sys.exit()
comm =MPI.COMM_WORLD
nproc = comm.Get_size()
f = h5py.File(filename, 'r', driver='mpio', comm=MPI.COMM_WORLD)
rank = comm.Get_rank()
#open the dataset, get the handle, shape
dset = f[dataset]


length_x = dset.shape[0]
length_y = dset.shape[1]
if rank==0:
   print (length_x,length_y)

f.atomic = False # for better performance

#devide the workload, 
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
ele_size=dset.dtype.itemsize
# Do the independent I/O
if colr==0: 
 temp[start:end,:] = dset[start:end,:]
else:
# Do the collective I/O
 with dset.collective:
   temp[start:end,:] = dset[start:end,:]
comm.Barrier()
#print "rank: ",rank,"\n", temp[start:end,:]
timeend=MPI.Wtime()
if rank==0:
    if colr==0:
     print "Independent read time: %f seconds" %(timeend-timestart)
    else:
     print "Collective read time: %f seconds" %(timeend-timestart)
    print "Data dimension x: %d y: %d" %(length_x, length_y)
    print "Data size: %d bytes" % (length_x*length_y*ele_size)
    print "Number of processes: %d" %nproc
f.close()
