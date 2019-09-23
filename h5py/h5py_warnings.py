# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    This module contains the warning classes for h5py. These classes are part of
    the public API of h5py, and should be imported from this module.
"""
from importlib import import_module


class H5pyWarning(UserWarning):
    pass


class H5pyDeprecationWarning(H5pyWarning,DeprecationWarning):
    pass


class ModuleWrapper(object):
    def __init__(self, mod):
        self._imported = False
        self._mod = mod

    def __getattr__(self, attr):
        if not self._imported:
            self._mod = self._import()
        return getattr(self._mod, attr)

    def _import(self):
        return import_module(self._mod)
