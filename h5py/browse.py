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

import cmd
import os
import posixpath
from utils_hl import hbasename

from h5py.highlevel import File, Group, Dataset, Datatype
from h5py import h5g

class _H5Browser(cmd.Cmd):

    """
        HDF5 file browser class which holds state between sessions.
    """

    def __init__(self):
        """ Create a new browser instance.
        """
        cmd.Cmd.__init__(self)
        self.path = '/'
        self.known_paths = {}
        self.file = None #: Holds a File instance while executing, the file name otherwise.

    def __call__(self, what=None, mode='r', importdict=None):
        """ Browse the file, putting any imported names into importdict. """

        if what is None:
            if self.file is None:
                raise ValueError("Either a file name or File object must be supplied.")
            else:
                self.file = File(self.file, mode=mode)

        elif isinstance(what, File):
            self.file = what

        elif isinstance(what, str):
            self.file = File(what, mode=mode)

        else:
            raise ValueError("Only a string file name or an File object may be supplied.")

        # Now self.file is a File object, no matter what

        try:
            self.path = self.known_paths[os.path.abspath(self.file.name)]
        except KeyError:
            self.path = '/'

        self.importdict = importdict
        self.cmdloop('Browsing "%s". Type "help" for commands, "exit" to exit.' % os.path.basename(self.file.name))
        self.importdict = None  # don't hold a reference to this between browse sessions

        self.known_paths[os.path.abspath(self.file.name)] = self.path
        self.file = self.file.name

    def _error(self, msg):
        print "Error: "+str(msg)

    def abspath(self, path):
        """ Correctly interpret the given path fragment, relative to the
            current path.
        """
        apath = posixpath.join(self.path, path)
        apath = posixpath.normpath(apath)
        return apath

    def get_candidates(path, criterion=lambda grp, name: True):
        """ Get a list of candidates, in the group pointed to by
            "path", which satisfy a particular criterion.
        """
        pass

    def do_exit(self, line):
        return True

    def do_EOF(self, line):
        return True

    def do_pwd(self, line):
        print self.path

    def do_cd(self, line):
        path = line.strip()
        if path == '': path = '/'
        path = self.abspath(path)
        dname = posixpath.dirname(path)
        bname = posixpath.basename(path)
        print dname, bname
        try:
            pgrp = self.file[dname]
            if bname != '' and not pgrp.id.get_objinfo(bname).type == h5g.GROUP:
                self._error('"%s" is not an HDF5 group' % bname)
            else:
                self.path = path
        except:
            self._error('Can\'t open group "%s"' % path)

    def complete_cd(self, text, line, begidx, endidx):
        text = text.strip()
        grpname = posixpath.join(self.path,posixpath.dirname(text))
        targetname = posixpath.basename(text)

        try:
            grp = self.file[grpname]
            return [posixpath.join(grpname,x) for x in grp \
                        if x.find(targetname) == 0 and \
                        grp.id.get_objinfo(x).type == h5g.GROUP]
        except:
            return []

    def do_ls(self, line):
        """ List contents of the specified group, or this one """

        line = line.strip()
        if line == '':
            grpname = self.path
        else:
            grpname = posixpath.join(self.path, line)

        try:
            grp = self.file[grpname]
            for name in grp:
                print name
        except:
            self._error('Can\'t list contents of group "%s"' % hbasename(grpname))

    def complete_ls(self, *args):
        return self.complete_cd(*args)













