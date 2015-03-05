Build instructions for h5py on Windows:

Build h5py in the normal fashion, except you are required to provide
both the --hdf5 and --hdf5-version arguments to setup.py.

Build HDF5 for distribution with a single command, using the pavement
file in this directory.  You will need to install paver first.

CMake 2.8 (NOT 3.0 or higher), and a SVN client, must be installed and
on the path.

To build HDF5 with Visual Studio 2008 (required for Python 2.6, 2.7 and 3.2):

  paver build_2008
  
To build with Visual Studio 2010 (required for Python 3.3):

  paver build_2010
  
These commands will each produce a zip file containing the appropriate
build of HDF5.  Unpack them and supply the appropriate directory to --hdf5.

Check pavement.py for the current HDF5 version to pass to --hdf5-version.