Build instructions for h5py on Windows

1. Building HDF5

  Because of the way the SZIP license is written, we cannot
  distribute the build of HDF5 provided by the HDF Group.
  The official builds of h5py include the SZIP decompressor only.
  Additionally, to avoid a runtime conflict with PyTables or
  other programs using HDF5, we have to rename the DLL.
  
  All of this is handled by the CMake cacheinit script in this
  directory.  To build HDF5, download the source code from
  hdfgroup.org and follow the Windows build instructions.
  Provide the h5py cacheinit file when required.
  
  Finally, create a ZIP distribution by using CPACK:
  
  cpack -G ZIP
  
  Unzip this file to c:\some\path; it should have directories
  like c:\some\path\bin and c:\some\path\lib.
  
2. Building h5py

  You will need Python, NumPy and Cython installed.  On Windows,
  the path to the HDF5 install directory must be provided.  You
  must also manually specify the version of HDF5:
  
  python setup.py --build --hdf5=c:\some\path --hdf5-version=1.8.12
  
3. Run tests

  Run the unit tests to make sure h5py has been properly built:
  
  python setup.py test