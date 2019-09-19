# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Demonstrates use of h5py in a multi-threaded GUI program.

    In a perfect world, multi-threaded programs would practice strict
    separation of tasks, with separate threads for HDF5, user interface,
    processing, etc, communicating via queues.  In the real world, shared
    state is frequently encountered, especially in the world of GUIs.  It's
    quite common to initialize a shared resource (in this case an HDF5 file),
    and pass it around between threads.  One must then be careful to regulate
    access using locks, to ensure that each thread sees the file in a
    consistent fashion.

    This program demonstrates how to use h5py in a medium-sized
    "shared-state" threading application.  Two threads exist: a GUI thread
    (Tkinter) which takes user input and displays results, and a calculation
    thread which is used to perform computation in the background, leaving
    the GUI responsive to user input.

    The computation thread calculates portions of the Mandelbrot set and
    stores them in an HDF5 file.  The visualization/control thread reads
    datasets from the same file and displays them using matplotlib.
"""

import tkinter as tk
import threading

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import h5py


file_lock = threading.RLock()  # Protects the file from concurrent access

t = None  # We'll use this to store the active computation thread

class ComputeThread(threading.Thread):

    """
        Computes a slice of the Mandelbrot set, and saves it to the HDF5 file.
    """

    def __init__(self, f, shape, escape, startcoords, extent, eventcall):
        """ Set up a computation thread.

        f: HDF5 File object
        shape: 2-tuple (NX, NY)
        escape: Integer giving max iterations to escape
        start: Complex number giving initial location on the plane
        extent: Complex number giving calculation extent on the plane
        """
        self.f = f
        self.shape = shape
        self.escape = escape
        self.startcoords = startcoords
        self.extent = extent
        self.eventcall = eventcall

        threading.Thread.__init__(self)

    def run(self):
        """ Perform computations and record the result to file """

        nx, ny = self.shape

        arr = np.ndarray((nx,ny), dtype='i')

        xincr = self.extent.real/nx
        yincr = self.extent.imag/ny

        def compute_escape(pos, escape):
            """ Compute the number of steps required to escape """
            z = 0+0j;
            for i in range(escape):
                z = z**2 + pos
                if abs(z) > 2:
                    break
            return i

        for x in range(nx):
            if x%25 == 0: print("Computing row %d" % x)
            for y in range(ny):
                pos = self.startcoords + complex(x*xincr, y*yincr)
                arr[x,y] = compute_escape(pos, self.escape)

        with file_lock:
            dsname = "slice%03d" % len(self.f)
            dset = self.f.create_dataset(dsname, (nx, ny), 'i')
            dset.attrs['shape'] = self.shape
            dset.attrs['start'] = self.startcoords
            dset.attrs['extent'] = self.extent
            dset.attrs['escape'] = self.escape
            dset[...] = arr

        print("Calculation for %s done" % dsname)

        self.eventcall()

class ComputeWidget(object):

    """
        Responsible for input widgets, and starting new computation threads.
    """

    def __init__(self, f, master, eventcall):

        self.f = f

        self.eventcall = eventcall

        self.mainframe = tk.Frame(master=master)

        entryframe = tk.Frame(master=self.mainframe)

        nxlabel = tk.Label(entryframe, text="NX")
        nylabel = tk.Label(entryframe, text="NY")
        escapelabel = tk.Label(entryframe, text="Escape")
        startxlabel = tk.Label(entryframe, text="Start X")
        startylabel = tk.Label(entryframe, text="Start Y")
        extentxlabel = tk.Label(entryframe, text="Extent X")
        extentylabel = tk.Label(entryframe, text="Extent Y")

        self.nxfield = tk.Entry(entryframe)
        self.nyfield = tk.Entry(entryframe)
        self.escapefield = tk.Entry(entryframe)
        self.startxfield = tk.Entry(entryframe)
        self.startyfield = tk.Entry(entryframe)
        self.extentxfield = tk.Entry(entryframe)
        self.extentyfield = tk.Entry(entryframe)

        nxlabel.grid(row=0, column=0, sticky=tk.E)
        nylabel.grid(row=1, column=0, sticky=tk.E)
        escapelabel.grid(row=2, column=0, sticky=tk.E)
        startxlabel.grid(row=3, column=0, sticky=tk.E)
        startylabel.grid(row=4, column=0, sticky=tk.E)
        extentxlabel.grid(row=5, column=0, sticky=tk.E)
        extentylabel.grid(row=6, column=0, sticky=tk.E)

        self.nxfield.grid(row=0, column=1)
        self.nyfield.grid(row=1, column=1)
        self.escapefield.grid(row=2, column=1)
        self.startxfield.grid(row=3, column=1)
        self.startyfield.grid(row=4, column=1)
        self.extentxfield.grid(row=5, column=1)
        self.extentyfield.grid(row=6, column=1)

        entryframe.grid(row=0, rowspan=2, column=0)

        self.suggestbutton = tk.Button(master=self.mainframe, text="Suggest", command=self.suggest)
        self.computebutton = tk.Button(master=self.mainframe, text="Compute", command=self.compute)

        self.suggestbutton.grid(row=0, column=1)
        self.computebutton.grid(row=1, column=1)

        self.suggest = 0

    def compute(self, *args):
        """ Validate input and start calculation thread.

        We use a global variable "t" to store the current thread, to make
        sure old threads are properly joined before they are discarded.
        """
        global t

        try:
            nx = int(self.nxfield.get())
            ny = int(self.nyfield.get())
            escape = int(self.escapefield.get())
            start = complex(float(self.startxfield.get()), float(self.startyfield.get()))
            extent = complex(float(self.extentxfield.get()), float(self.extentyfield.get()))
            if (nx<=0) or (ny<=0) or (escape<=0):
                raise ValueError("NX, NY and ESCAPE must be positive")
            if abs(extent)==0:
                raise ValueError("Extent must be finite")
        except (ValueError, TypeError) as e:
            print(e)
            return

        if t is not None:
            t.join()

        t = ComputeThread(self.f, (nx,ny), escape, start, extent, self.eventcall)
        t.start()

    def suggest(self, *args):
        """ Populate the input fields with interesting locations """

        suggestions = [(200,200,50, -2, -1, 3, 2),
                       (500, 500, 200, 0.110, -0.680, 0.05, 0.05),
                       (200, 200, 1000, -0.16070135-5e-8, 1.0375665-5e-8, 1e-7, 1e-7),
                       (500, 500, 100, -1, 0, 0.5, 0.5)]

        for entry, val in zip((self.nxfield, self.nyfield, self.escapefield,
                self.startxfield, self.startyfield, self.extentxfield,
                self.extentyfield), suggestions[self.suggest]):
            entry.delete(0, 999)
            entry.insert(0, repr(val))

        self.suggest = (self.suggest+1)%len(suggestions)


class ViewWidget(object):

    """
        Draws images using the datasets recorded in the HDF5 file.  Also
        provides widgets to pick which dataset is displayed.
    """

    def __init__(self, f, master):

        self.f = f

        self.mainframe = tk.Frame(master=master)
        self.lbutton = tk.Button(self.mainframe, text="<= Back", command=self.back)
        self.rbutton = tk.Button(self.mainframe, text="Next =>", command=self.forward)
        self.loclabel = tk.Label(self.mainframe, text='To start, enter values and click "compute"')
        self.infolabel = tk.Label(self.mainframe, text='Or, click the "suggest" button for interesting locations')

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.mainframe)
        self.canvas.draw_idle()

        self.loclabel.grid(row=0, column=1)
        self.infolabel.grid(row=1, column=1)
        self.lbutton.grid(row=2, column=0)
        self.canvas.get_tk_widget().grid(row=2, column=1)
        self.rbutton.grid(row=2, column=2)

        self.index = 0

        self.jumptolast()

    def draw_fractal(self):
        """ Read a dataset from the HDF5 file and display it """

        with file_lock:
            name = list(self.f.keys())[self.index]
            dset = self.f[name]
            arr = dset[...]
            start = dset.attrs['start']
            extent = dset.attrs['extent']
            self.loclabel["text"] = 'Displaying dataset "%s" (%d of %d)' % (dset.name, self.index+1, len(self.f))
            self.infolabel["text"] = "%(shape)s pixels, starts at %(start)s, extent %(extent)s" % dset.attrs

        self.plot.clear()
        self.plot.imshow(arr.transpose(), cmap='jet', aspect='auto', origin='lower',
                         extent=(start.real, (start.real+extent.real),
                                 start.imag, (start.imag+extent.imag)))
        self.canvas.draw_idle()

    def back(self):
        """ Go to the previous dataset (in ASCII order) """
        if self.index == 0:
            print("Can't go back")
            return
        self.index -= 1
        self.draw_fractal()

    def forward(self):
        """ Go to the next dataset (in ASCII order) """
        if self.index == (len(self.f)-1):
            print("Can't go forward")
            return
        self.index += 1
        self.draw_fractal()

    def jumptolast(self,*args):
        """ Jump to the last (ASCII order) dataset and display it """
        with file_lock:
            if len(self.f) == 0:
                print("can't jump to last (no datasets)")
                return
            index = len(self.f)-1
        self.index = index
        self.draw_fractal()


if __name__ == '__main__':

    f = h5py.File('mandelbrot_gui.hdf5', 'a')

    root = tk.Tk()

    display = ViewWidget(f, root)

    root.bind("<<FractalEvent>>", display.jumptolast)
    def callback():
        root.event_generate("<<FractalEvent>>")
    compute = ComputeWidget(f, root, callback)

    display.mainframe.grid(row=0, column=0)
    compute.mainframe.grid(row=1, column=0)

    try:
        root.mainloop()
    finally:
        if t is not None:
            t.join()
        f.close()
