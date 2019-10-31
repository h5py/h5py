# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Demonstrates how to use h5py with the multiprocessing module.

    This module implements a simple multi-process program to generate
    Mandelbrot set images.  It uses a process pool to do the computations,
    and a single process to save the results to file.

    Importantly, only one process actually reads/writes the HDF5 file.
    Remember that when a process is fork()ed, the child inherits the HDF5
    state from its parent, which can be dangerous if you already have a file
    open.  Trying to interact with the same file on disk from multiple
    processes results in undefined behavior.

    If matplotlib is available, the program will read from the HDF5 file and
    display an image of the fractal in a window.  To re-run the calculation,
    delete the file "mandelbrot.hdf5".

"""

import numpy as np
import multiprocessing as mp
import h5py

# === Parameters for Mandelbrot calculation ===================================

NX = 512
NY = 512
ESCAPE = 1000

XSTART = -0.16070135 - 5e-8
YSTART =  1.0375665 -5e-8
XEXTENT = 1.0E-7
YEXTENT = 1.0E-7

xincr = XEXTENT*1.0/NX
yincr = YEXTENT*1.0/NY

# === Functions to compute set ================================================

def compute_escape(pos):
    """ Compute the number of steps required to escape from a point on the
    complex plane """
    z = 0+0j;
    for i in range(ESCAPE):
        z = z**2 + pos
        if abs(z) > 2:
            break
    return i

def compute_row(xpos):
    """ Compute a 1-D array containing escape step counts for each y-position.
    """
    a = np.ndarray((NY,), dtype='i')
    for y in range(NY):
        pos = complex(XSTART,YSTART) + complex(xpos, y*yincr)
        a[y] = compute_escape(pos)
    return a

# === Functions to run process pool & visualize ===============================

def run_calculation():
    """ Begin multi-process calculation, and save to file """

    print("Creating %d-process pool" % mp.cpu_count())

    pool = mp.Pool(mp.cpu_count())

    f = h5py.File('mandelbrot.hdf5','w')

    print("Creating output dataset with shape %s x %s" % (NX, NY))

    dset = f.create_dataset('mandelbrot', (NX,NY), 'i')
    dset.attrs['XSTART'] = XSTART
    dset.attrs['YSTART'] = YSTART
    dset.attrs['XEXTENT'] = XEXTENT
    dset.attrs['YEXTENT'] = YEXTENT

    result = pool.imap(compute_row, (x*xincr for x in range(NX)))

    for idx, arr in enumerate(result):
        if idx%25 == 0: print("Recording row %s" % idx)
        dset[idx] = arr

    print("Closing HDF5 file")

    f.close()

    print("Shutting down process pool")

    pool.close()
    pool.join()

def visualize_file():
    """ Open the HDF5 file and display the result """
    try:
        import pylab as p
    except ImportError:
        print("Whoops! Matplotlib is required to view the fractal.")
        raise

    f = h5py.File('mandelbrot.hdf5','r')
    dset = f['mandelbrot']
    a = dset[...]
    p.imshow(a.transpose())

    print("Displaying fractal. Close window to exit program.")
    try:
        p.show()
    finally:
        f.close()

if __name__ == '__main__':
    if not h5py.is_hdf5('mandelbrot.hdf5'):
        run_calculation()
    else:
        print('Fractal found in "mandelbrot.hdf5". Delete file to re-run calculation.')
    visualize_file()
