# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements high-level support for HDF5 file objects.
"""

from __future__ import absolute_import

import sys
import os

from .compat import fspath
from .compat import fsencode
from .compat import fsdecode

import six

from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version

mpi = h5.get_config().mpi
hdf5_version = version.hdf5_version_tuple[0:3]

swmr_support = False
if hdf5_version >= h5.get_config().swmr_min_hdf5_version:
    swmr_support = True

if mpi:
    import mpi4py

libver_dict = {'earliest': h5f.LIBVER_EARLIEST, 'latest': h5f.LIBVER_LATEST}
libver_dict_r = dict((y, x) for x, y in six.iteritems(libver_dict))


def make_fapl(driver, libver, **kwds):
    """ Set up a file access property list """
    plist = h5p.create(h5p.FILE_ACCESS)

    if libver is not None:
        if libver in libver_dict:
            low = libver_dict[libver]
            high = h5f.LIBVER_LATEST
        else:
            low, high = (libver_dict[x] for x in libver)
        plist.set_libver_bounds(low, high)

    if driver is None or (driver == 'windows' and sys.platform == 'win32'):
        # Prevent swallowing unused key arguments
        if kwds:
            msg = "'{key}' is an invalid keyword argument for this function" \
                  .format(key=next(iter(kwds)))
            raise TypeError(msg)
        return plist

    if driver == 'sec2':
        plist.set_fapl_sec2(**kwds)
    elif driver == 'stdio':
        plist.set_fapl_stdio(**kwds)
    elif driver == 'core':
        plist.set_fapl_core(**kwds)
    elif driver == 'family':
        plist.set_fapl_family(memb_fapl=plist.copy(), **kwds)
    elif driver == 'mpio':
        kwds.setdefault('info', mpi4py.MPI.Info())
        plist.set_fapl_mpio(**kwds)
    else:
        raise ValueError('Unknown driver type "%s"' % driver)

    return plist


def make_fid(name, mode, userblock_size, fapl, fcpl=None, swmr=False):
    """ Get a new FileID by opening or creating a file.
    Also validates mode argument."""

    if userblock_size is not None:
        if mode in ('r', 'r+'):
            raise ValueError("User block may only be specified "
                             "when creating a file")
        try:
            userblock_size = int(userblock_size)
        except (TypeError, ValueError):
            raise ValueError("User block size must be an integer")
        if fcpl is None:
            fcpl = h5p.create(h5p.FILE_CREATE)
        fcpl.set_userblock(userblock_size)

    if mode == 'r':
        flags = h5f.ACC_RDONLY
        if swmr and swmr_support:
            flags |= h5f.ACC_SWMR_READ
        fid = h5f.open(name, flags, fapl=fapl)
    elif mode == 'r+':
        fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
    elif mode in ['w-', 'x']:
        fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
    elif mode == 'w':
        fid = h5f.create(name, h5f.ACC_TRUNC, fapl=fapl, fcpl=fcpl)
    elif mode == 'a':
        # Open in append mode (read/write).
        # If that fails, create a new file only if it won't clobber an
        # existing one (ACC_EXCL)
        try:
            fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
        except IOError:
            fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
    elif mode is None:
        # Try to open in append mode (read/write).
        # If that fails, try readonly, and finally create a new file only
        # if it won't clobber an existing file (ACC_EXCL).
        try:
            fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
        except IOError:
            try:
                fid = h5f.open(name, h5f.ACC_RDONLY, fapl=fapl)
            except IOError:
                fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
    else:
        raise ValueError("Invalid mode; must be one of r, r+, w, w-, x, a")

    try:
        if userblock_size is not None:
            existing_fcpl = fid.get_create_plist()
            if existing_fcpl.get_userblock() != userblock_size:
                raise ValueError("Requested userblock size (%d) does not match that of existing file (%d)" % (userblock_size, existing_fcpl.get_userblock()))
    except:
        fid.close()
        raise

    return fid


class File(Group):

    """
        Represents an HDF5 file.
    """

    @property
    @with_phil
    def attrs(self):
        """ Attributes attached to this object """
        # hdf5 complains that a file identifier is an invalid location for an
        # attribute. Instead of self, pass the root group to AttributeManager:
        from . import attrs
        return attrs.AttributeManager(self['/'])

    @property
    @with_phil
    def filename(self):
        """File name on disk"""
        return fsdecode(h5f.get_name(self.fid))

    @property
    @with_phil
    def driver(self):
        """Low-level HDF5 file driver used to open file"""
        drivers = {h5fd.SEC2: 'sec2', h5fd.STDIO: 'stdio',
                   h5fd.CORE: 'core', h5fd.FAMILY: 'family',
                   h5fd.WINDOWS: 'windows', h5fd.MPIO: 'mpio',
                   h5fd.MPIPOSIX: 'mpiposix'}
        return drivers.get(self.fid.get_access_plist().get_driver(), 'unknown')

    @property
    @with_phil
    def mode(self):
        """ Python mode used to open file """
        return {h5f.ACC_RDONLY: 'r',
                h5f.ACC_RDWR: 'r+'}.get(self.fid.get_intent())

    @property
    @with_phil
    def fid(self):
        """File ID (backwards compatibility) """
        return self.id

    @property
    @with_phil
    def libver(self):
        """File format version bounds (2-tuple: low, high)"""
        bounds = self.id.get_access_plist().get_libver_bounds()
        return tuple(libver_dict_r[x] for x in bounds)

    @property
    @with_phil
    def userblock_size(self):
        """ User block size (in bytes) """
        fcpl = self.fid.get_create_plist()
        return fcpl.get_userblock()


    if mpi and hdf5_version >= (1, 8, 9):

        @property
        @with_phil
        def atomic(self):
            """ Set/get MPI-IO atomic mode
            """
            return self.id.get_mpi_atomicity()

        @atomic.setter
        @with_phil
        def atomic(self, value):
            # pylint: disable=missing-docstring
            self.id.set_mpi_atomicity(value)

    if swmr_support:
        @property
        def swmr_mode(self):
            """ Controls single-writer multiple-reader mode """
            return self._swmr_mode

        @swmr_mode.setter
        @with_phil
        def swmr_mode(self, value):
            # pylint: disable=missing-docstring
            if value:
                self.id.start_swmr_write()
                self._swmr_mode = True
            else:
                raise ValueError("It is not possible to forcibly switch SWMR mode off.")

    def __init__(self, name, mode=None, driver=None,
                 libver=None, userblock_size=None, swmr=False, **kwds):
        """Create a new file object.

        See the h5py user guide for a detailed explanation of the options.

        name
            Name of the file on disk.  Note: for files created with the 'core'
            driver, HDF5 still requires this be non-empty.
        mode
            r        Readonly, file must exist
            r+       Read/write, file must exist
            w        Create file, truncate if exists
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise (default)
        driver
            Name of the driver to use.  Legal values are None (default,
            recommended), 'core', 'sec2', 'stdio', 'mpio'.
        libver
            Library version bounds.  Currently only the strings 'earliest'
            and 'latest' are defined.
        userblock
            Desired size of user block.  Only allowed when creating a new
            file (mode w, w- or x).
        swmr
            Open the file in SWMR read mode. Only used when mode = 'r'.
        Additional keywords
            Passed on to the selected file driver.
        """
        if swmr and not swmr_support:
            raise ValueError("The SWMR feature is not available in this version of the HDF5 library")

        with phil:
            if isinstance(name, _objects.ObjectID):
                fid = h5i.get_file_id(name)
            else:
                name = fsencode(fspath(name))

                fapl = make_fapl(driver, libver, **kwds)
                fid = make_fid(name, mode, userblock_size, fapl, swmr=swmr)

                if swmr_support:
                    self._swmr_mode = False
                    if swmr and mode == 'r':
                        self._swmr_mode = True

            Group.__init__(self, fid)

    def close(self):
        """ Close the file.  All open objects become invalid """
        with phil:
            # We have to explicitly murder all open objects related to the file

            # Close file-resident objects first, then the files.
            # Otherwise we get errors in MPI mode.
            id_list = h5f.get_obj_ids(self.id, ~h5f.OBJ_FILE)
            file_list = h5f.get_obj_ids(self.id, h5f.OBJ_FILE)

            id_list = [x for x in id_list if h5i.get_file_id(x).id == self.id.id]
            file_list = [x for x in file_list if h5i.get_file_id(x).id == self.id.id]

            for id_ in id_list:
                while id_.valid:
                    h5i.dec_ref(id_)

            for id_ in file_list:
                while id_.valid:
                    h5i.dec_ref(id_)

            self.id.close()
            _objects.nonlocal_close()

    def flush(self):
        """ Tell the HDF5 library to flush its buffers.
        """
        with phil:
            h5f.flush(self.fid)

    @with_phil
    def __enter__(self):
        return self

    @with_phil
    def __exit__(self, *args):
        if self.id:
            self.close()

    @with_phil
    def __repr__(self):
        if not self.id:
            r = six.u('<Closed HDF5 file>')
        else:
            # Filename has to be forced to Unicode if it comes back bytes
            # Mode is always a "native" string
            filename = self.filename
            if isinstance(filename, bytes):  # Can't decode fname
                filename = filename.decode('utf8', 'replace')
            r = six.u('<HDF5 file "%s" (mode %s)>') % (os.path.basename(filename),
                                                 self.mode)

        if six.PY2:
            return r.encode('utf8')
        return r
