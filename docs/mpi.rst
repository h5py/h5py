.. _parallel:

Parallel HDF5
=============

Read-only parallel access to HDF5 files works with no special preparation:
each process should open the file independently and read data normally
(avoid opening the file and then forking).

`Parallel HDF5 <https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5>`_ is a
feature built on MPI which also supports *writing* an HDF5 file in parallel.
To use this, both HDF5 and h5py must be compiled with MPI support turned on,
as described below.

How does Parallel HDF5 work?
----------------------------

Parallel HDF5 is a configuration of the HDF5 library which lets you share
open files across multiple parallel processes.  It uses the MPI (Message
Passing Interface) standard for interprocess communication.  Consequently,
when using Parallel HDF5 from Python, your application will also have to use
the MPI library.

This is accomplished through the `mpi4py <http://mpi4py.scipy.org/>`_ Python package, which provides
excellent, complete Python bindings for MPI.  Here's an example
"Hello World" using ``mpi4py``::

    >>> from mpi4py import MPI
    >>> print("Hello World (from process %d)" % MPI.COMM_WORLD.Get_rank())

To run an MPI-based parallel program, use the ``mpiexec`` program to launch
several parallel instances of Python::

    $ mpiexec -n 4 python demo.py
    Hello World (from process 1)
    Hello World (from process 2)
    Hello World (from process 3)
    Hello World (from process 0)

The ``mpi4py`` package includes all kinds of mechanisms to share data between
processes, synchronize, etc.  It's a different flavor of parallelism than,
say, threads or ``multiprocessing``, but easy to get used to.

Check out the `mpi4py web site <http://mpi4py.scipy.org/>`_ for more information
and a great tutorial.


Building against Parallel HDF5
------------------------------

HDF5 must be built with at least the following options::

    $./configure --enable-parallel --enable-shared

Note that ``--enable-shared`` is required.

Often, a "parallel" version of HDF5 will be available through your package
manager.  You can check to see what build options were used by using the
program ``h5cc``::

    $ h5cc -showconfig

Once you've got a Parallel-enabled build of HDF5, h5py has to be compiled in
"MPI mode".  Set your default compiler to the ``mpicc`` wrapper
and build h5py with the ``HDF5_MPI`` environment variable::

    $ export CC=mpicc
    $ export HDF5_MPI="ON"
    $ export HDF5_DIR="/path/to/parallel/hdf5"  # If this isn't found by default
    $ pip install .


Using Parallel HDF5 from h5py
-----------------------------

The parallel features of HDF5 are mostly transparent.  To open a file shared
across multiple processes, use the ``mpio`` file driver.  Here's an example
program which opens a file, creates a single dataset and fills it with the
process ID::


    from mpi4py import MPI
    import h5py

    rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

    f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)

    dset = f.create_dataset('test', (4,), dtype='i')
    dset[rank] = rank

    f.close()

Run the program::

    $ mpiexec -n 4 python demo2.py

Looking at the file with ``h5dump``::

    $ h5dump parallel_test.hdf5
    HDF5 "parallel_test.hdf5" {
    GROUP "/" {
       DATASET "test" {
          DATATYPE  H5T_STD_I32LE
          DATASPACE  SIMPLE { ( 4 ) / ( 4 ) }
          DATA {
          (0): 0, 1, 2, 3
          }
       }
    }
    }

Collective versus independent operations
----------------------------------------

MPI-based programs work by launching many instances of the Python interpreter,
each of which runs your script.  There are certain requirements imposed on
what each process can do.  Certain operations in HDF5, for example, anything
which modifies the file metadata, must be performed by all processes.  Other
operations, for example, writing data to a dataset, can be performed by some
processes and not others.

These two classes are called *collective* and *independent* operations.  Anything
which modifies the *structure* or metadata of a file must be done collectively.
For example, when creating a group, each process must participate::

    >>> grp = f.create_group('x')  # right

    >>> if rank == 1:
    ...     grp = f.create_group('x')   # wrong; all processes must do this

On the other hand, writing data to a dataset can be done independently::

    >>> if rank > 2:
    ...     dset[rank] = 42   # this is fine


MPI atomic mode
---------------

HDF5 versions 1.8.9+ support the MPI "atomic" file access mode, which trades
speed for more stringent consistency requirements.  Once you've opened a
file with the ``mpio`` driver, you can place it in atomic mode using the
settable ``atomic`` property::

    >>> f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
    >>> f.atomic = True


More information
----------------

Parallel HDF5 is a new feature in h5py.  If you have any questions, feel free to
ask on the mailing list (h5py at google groups).  We welcome bug reports,
enhancements and general inquiries.
