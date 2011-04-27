import weakref
import sys
import os

from base import HLObject
from group import Group
from h5py import h5f, h5p, h5i, h5fd
import shared

def make_fapl(driver,**kwds):
    """ Set up a file access property list """
    plist = h5p.create(h5p.FILE_ACCESS)
    plist.set_fclose_degree(h5f.CLOSE_STRONG)

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

def make_fid(name, mode, plist):
    """ Get a new FileID by opening or creating a file.
    Also validates mode argument."""
    if mode == 'r':
        fid = h5f.open(name, h5f.ACC_RDONLY, fapl=plist)
    elif mode == 'r+':
        fid = h5f.open(name, h5f.ACC_RDWR, fapl=plist)
    elif mode == 'w-':
        fid = h5f.create(name, h5f.ACC_EXCL, fapl=plist)
    elif mode == 'w':
        fid = h5f.create(name, h5f.ACC_TRUNC, fapl=plist)
    elif mode == 'a' or mode is None:
        try:
            fid = h5f.open(name, h5f.ACC_RDWR, fapl=plist)
        except IOError:
            fid = h5f.create(name, h5f.ACC_EXCL, fapl=plist)
    else:
        raise ValueError("Invalid mode; must be one of r, r+, w, w-, a")
    return fid

class File(Group):

    """
        Represents an HDF5 file
    """

    @property
    def filename(self):
        """File name on disk"""
        name = h5f.get_name(self.fid)
        # Note the exception can happen in one of two ways:
        # 1. The name doesn't comply with the file system encoding;
        #    return the raw byte string
        # 2. The name can't be encoded down to ASCII; return it as
        #    a Unicode string object
        try:
            name = name.decode(sys.getfilesystemencoding())
            return name.encode('ascii')
        except (UnicodeError, LookupError):
            return name

    
    @property
    def driver(self):
        """Low-level HDF5 file driver used to open file"""
        drivers = {h5fd.SEC2: 'sec2', h5fd.STDIO: 'stdio',
                   h5fd.CORE: 'core', h5fd.FAMILY: 'family',
                   h5fd.WINDOWS: 'windows'}
        return drivers.get(self.fid.get_access_plist().get_driver(), 'unknown')

    @shared.shared
    def mode(self, sc):
        """ Python mode used to open file """
        mode = sc.get('mode')
        if mode is None:
            mode = {h5f.ACC_RDONLY: 'r', h5f.ACC_RDWR: 'r+'}.get(self.fid.get_intent())
            sc['mode'] = mode
        return mode

    @property
    def fid(self):
        """File ID (backwards compatibility) """
        return self.id

    def __init__(self, name, mode=None, driver=None, **kwds):
        """ Create a new file object """
        if isinstance(name, HLObject):
            fid = h5i.get_file_id(name.id)
        else:
            try:
                # If the byte string doesn't match the default encoding, just
                # pass it on as-is.  Note Unicode objects can always be encoded.
                name = name.encode(sys.getfilesystemencoding())
            except (UnicodeError, LookupError):
                pass
            fapl = make_fapl(driver,**kwds)
            fid = make_fid(name, mode, fapl)
        Group.__init__(self, None, None, bind=fid)
        shared.setval(self, 'mode', mode)

    def close(self):
        """ Close the file.  All open objects become invalid """
        shared.wipe(self)
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



