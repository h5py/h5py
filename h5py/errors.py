#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

class H5Error(StandardError):
    pass

class ConversionError(H5Error):
    pass

class H5LibraryError(H5Error):
    pass

class FileError(H5LibraryError):
    pass

class GroupError(H5LibraryError):
    pass

class DataspaceError(H5LibraryError):
    pass

class DatatypeError(H5LibraryError):
    pass

class DatasetError(H5LibraryError):
    pass

class PropertyError(H5LibraryError):
    pass

class H5AttributeError(H5LibraryError):
    pass

class FilterError(H5LibraryError):
    pass

class H5TypeError(H5LibraryError):
    pass

class IdentifierError(H5LibraryError):
    pass

class H5ReferenceError(H5LibraryError):
    pass



