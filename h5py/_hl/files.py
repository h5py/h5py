import weakref
import sys
import os

from .base import HLObject
from .group import Group
from h5py import h5f, h5p, h5i, h5fd, h5t, _objects

libver_dict = {'earliest': h5f.LIBVER_EARLIEST, 'latest': h5f.LIBVER_LATEST}
libver_dict_r = dict((y,x) for x, y in libver_dict.iteritems())

def make_fapl(driver,libver,**kwds):
    """ Set up a file access property list """
    plist = h5p.create(h5p.FILE_ACCESS)
    plist.set_fclose_degree(h5f.CLOSE_STRONG)

    if libver is not None:
        if libver in libver_dict:
            low = libver_dict[libver]
            high = h5f.LIBVER_LATEST
        else:
            low, high = (libver_dict[x] for x in libver)
        plist.set_libver_bounds(low, high)

    if driver is None or (driver=='windows' and sys.platform=='win32'):
        return plist

    if(driver=='sec2'):
        plist.set_fapl_sec2(**kwds)
    elif(driver=='stdio'):
        plist.set_fapl_stdio(**kwds)
    elif(driver=='core'):
        plist.set_fapl_core(**kwds)
    elif(driver=='family'):
        plist.set_fapl_family(memb_fapl=plist.copy(), **kwds)
    else:
        raise ValueError('Unknown driver type "%s"' % driver)

    return plist

def make_fid(name, mode, userblock_size, fapl):
    """ Get a new FileID by opening or creating a file.
    Also validates mode argument."""

    fcpl=None
    if userblock_size is not None:
        if mode in ('r', 'r+'):
            raise ValueError("User block may only be specified when creating a file")
        try:
            userblock_size = int(userblock_size)
        except (TypeError, ValueError):
            raise ValueError("User block size must be an integer")
        fcpl = h5p.create(h5p.FILE_CREATE)
        fcpl.set_userblock(userblock_size)

    if mode == 'r':
        fid = h5f.open(name, h5f.ACC_RDONLY, fapl=fapl)
    elif mode == 'r+':
        fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
    elif mode == 'w-':
        fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
    elif mode == 'w':
        fid = h5f.create(name, h5f.ACC_TRUNC, fapl=fapl, fcpl=fcpl)
    elif mode == 'a' or mode is None:
        try:
            fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)
            try:
                existing_fcpl = fid.get_create_plist()
                if userblock_size is not None and existing_fcpl.get_userblock() != userblock_size:
                    raise ValueError("Requested userblock size (%d) does not match that of existing file (%d)" % (userblock_size, existing_fcpl.get_userblock()))
            except:
                fid.close()
                raise
        except IOError:
            fid = h5f.create(name, h5f.ACC_EXCL, fapl=fapl, fcpl=fcpl)
    else:
        raise ValueError("Invalid mode; must be one of r, r+, w, w-, a")

    return fid

class File(Group):

    """
        Represents an HDF5 file.
    """

    @property
    def attrs(self):
        """ Attributes attached to this object """
        # hdf5 complains that a file identifier is an invalid location for an
        # attribute. Instead of self, pass the root group to AttributeManager:
        import attrs
        return attrs.AttributeManager(self['/'])

    @property
    def filename(self):
        """File name on disk"""
        name = h5f.get_name(self.fid)
        try:
            return name.decode(sys.getfilesystemencoding())
        except (UnicodeError, LookupError):
            return name

    @property
    def driver(self):
        """Low-level HDF5 file driver used to open file"""
        drivers = {h5fd.SEC2: 'sec2', h5fd.STDIO: 'stdio',
                   h5fd.CORE: 'core', h5fd.FAMILY: 'family',
                   h5fd.WINDOWS: 'windows'}
        return drivers.get(self.fid.get_access_plist().get_driver(), 'unknown')

    @property
    def mode(self):
        """ Python mode used to open file """
        return {h5f.ACC_RDONLY: 'r', h5f.ACC_RDWR: 'r+'}.get(self.fid.get_intent())

    @property
    def fid(self):
        """File ID (backwards compatibility) """
        return self.id

    @property
    def libver(self):
        """File format version bounds (2-tuple: low, high)"""
        bounds = self.id.get_access_plist().get_libver_bounds()
        return tuple(libver_dict_r[x] for x in bounds)

    @property
    def userblock_size(self):
        """ User block size (in bytes) """
        fcpl = self.fid.get_create_plist()
        return fcpl.get_userblock()

    def __init__(self, name, mode=None, driver=None, libver=None, userblock_size=None,
        **kwds):
        """Create a new file object.

        See the h5py user guide for a detailed explanation of the options.

        name
            Name of the file on disk.  Note: for files created with the 'core'
            driver, HDF5 still requires this be non-empty.
        driver
            Name of the driver to use.  Legal values are None (default,
            recommended), 'core', 'sec2' (UNIX), 'stdio'.
        libver
            Library version bounds.  Currently only the strings 'earliest'
            and 'latest' are defined.
        userblock
            Desired size of user block.  Only allowed when creating a new
            file (mode w or w-).
        Additional keywords
            Passed on to the selected file driver.
        """
        if isinstance(name, _objects.ObjectID):
            fid = h5i.get_file_id(name)
        else:
            try:
                # If the byte string doesn't match the default encoding, just
                # pass it on as-is.  Note Unicode objects can always be encoded.
                name = name.encode(sys.getfilesystemencoding())
            except (UnicodeError, LookupError):
                pass
            fapl = make_fapl(driver,libver,**kwds)
            fid = make_fid(name, mode, userblock_size, fapl)
        Group.__init__(self, fid)

    def close(self):
        """ Close the file.  All open objects become invalid """
        # TODO: find a way to square this with having issue 140
        # Not clearing shared state introduces a tiny memory leak, but
        # it goes like the number of files opened in a session.
        self.id.close()

    def flush(self):
        """ Tell the HDF5 library to flush its buffers.
        """
        h5f.flush(self.fid)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        if self.id:
            self.close()

    def __repr__(self):
        if not self.id:
            return "<Closed HDF5 file>"
        return '<HDF5 file "%s" (mode %s)>' % \
            (os.path.basename(self.filename), self.mode)



