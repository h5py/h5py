# This is the official h5py CMakeCache file.
# Please see README.txt in the "windows" directory of the h5py package for more info.

########################
# EXTERNAL cache entries
########################

SET (BUILD_SHARED_LIBS ON CACHE BOOL "Build Shared Libraries" FORCE)

SET (BUILD_TESTING ON CACHE BOOL "Build HDF5 Unit Testing" FORCE)

SET (HDF_PACKAGE_EXT "" CACHE STRING "Name of HDF package extension" FORCE)

SET (HDF5_EXTERNAL_LIB_PREFIX "h5py_" CACHE STRING "docstring" FORCE)

SET (HDF5_BUILD_CPP_LIB OFF CACHE BOOL "Build HDF5 C++ Library" FORCE)

SET (HDF5_BUILD_EXAMPLES OFF CACHE BOOL "Build HDF5 Library Examples" FORCE)

SET (HDF5_BUILD_FORTRAN OFF CACHE BOOL "Build FORTRAN support" FORCE)

SET (HDF5_ENABLE_F2003 OFF CACHE BOOL "Enable FORTRAN 2003 Standard" FORCE)

SET (HDF5_BUILD_HL_LIB ON CACHE BOOL "Build HIGH Level HDF5 Library" FORCE)

SET (HDF5_BUILD_TOOLS ON CACHE BOOL "Build HDF5 Tools" FORCE)

SET (HDF5_BUILD_GENERATORS OFF CACHE BOOL "Build Test Generators" FORCE)

SET (HDF5_ENABLE_Z_LIB_SUPPORT ON CACHE BOOL "Enable Zlib Filters" FORCE)

SET (HDF5_ENABLE_SZIP_SUPPORT ON CACHE BOOL "Use SZip Filter" FORCE)

SET (HDF5_ENABLE_SZIP_ENCODING OFF CACHE BOOL "Use SZip Encoding" FORCE)

SET (HDF5_ENABLE_HSIZET ON CACHE BOOL "Enable datasets larger than memory" FORCE)

SET (HDF5_ENABLE_UNSUPPORTED OFF CACHE BOOL "Enable unsupported combinations of configuration options" FORCE)

SET (HDF5_ENABLE_DEPRECATED_SYMBOLS ON CACHE BOOL "Enable deprecated public API symbols" FORCE)

SET (HDF5_ENABLE_DIRECT_VFD OFF CACHE BOOL "Build the Direct I/O Virtual File Driver" FORCE)

SET (HDF5_ENABLE_PARALLEL OFF CACHE BOOL "Enable parallel build (requires MPI)" FORCE)

SET (MPIEXEC_MAX_NUMPROCS "3" CACHE STRING "Minimum number of processes for HDF parallel tests" FORCE)

SET (HDF5_BUILD_PARALLEL_ALL OFF CACHE BOOL "Build Parallel Programs" FORCE)

SET (HDF5_ENABLE_COVERAGE OFF CACHE BOOL "Enable code coverage for Libraries and Programs" FORCE)

SET (HDF5_ENABLE_USING_MEMCHECKER OFF CACHE BOOL "Indicate that a memory checker is used" FORCE)

SET (HDF5_DISABLE_COMPILER_WARNINGS OFF CACHE BOOL "Disable compiler warnings" FORCE)

SET (HDF5_USE_FOLDERS ON CACHE BOOL "Enable folder grouping of projects in IDEs." FORCE)

SET (HDF5_USE_16_API_DEFAULT OFF CACHE BOOL "Use the HDF5 1.6.x API by default" FORCE)

SET (HDF5_ENABLE_THREADSAFE OFF CACHE BOOL "(WINDOWS)Enable Threadsafety" FORCE)

SET (HDF5_PACKAGE_EXTLIBS ON CACHE BOOL "(WINDOWS)CPACK - include external libraries" FORCE)

SET (HDF5_NO_PACKAGES OFF CACHE BOOL "CPACK - Disable packaging" FORCE)

SET (HDF5_ALLOW_EXTERNAL_SUPPORT "SVN" CACHE STRING "Allow External Library Building (NO SVN TGZ)" FORCE)
SET_PROPERTY(CACHE HDF5_ALLOW_EXTERNAL_SUPPORT PROPERTY STRINGS NO SVN TGZ)

SET (ZLIB_SVN_URL "http://svn.hdfgroup.uiuc.edu/zlib/trunk" CACHE STRING "Use ZLib from HDF repository" FORCE)

SET (SZIP_SVN_URL "http://svn.hdfgroup.uiuc.edu/szip/trunk" CACHE STRING "Use SZip from HDF repository" FORCE)

SET (ZLIB_TGZ_NAME "ZLib.tar.gz" CACHE STRING "Use ZLib from compressed file" FORCE)

SET (SZIP_TGZ_NAME "SZip.tar.gz" CACHE STRING "Use SZip from compressed file" FORCE)

SET (ZLIB_PACKAGE_NAME "zlib" CACHE STRING "Name of ZLIB package" FORCE)

SET (SZIP_PACKAGE_NAME "szip" CACHE STRING "Name of SZIP package" FORCE)
