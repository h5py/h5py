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

from h5 import get_error_string

class H5Error(StandardError):
    pass

class ConversionError(H5Error):
    pass

class H5LibraryError(H5Error):

    def __init__(self, *args, **kwds):
        arglist = list(args)
        if len(arglist) == 0:
            arglist = [""]
        arglist[0] = arglist[0] + "\n" + get_error_string()
        args = tuple(arglist)
        H5Error.__init__(self, *args, **kwds)

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





