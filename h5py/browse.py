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
from getopt import gnu_getopt
import os
from utils_hl import hbasename
from posixpath import join, basename, dirname, normpath, isabs

from h5py.highlevel import File, Group, Dataset, Datatype
from h5py import h5g

NAMES = {h5g.DATASET: "Dataset", h5g.GROUP: "Group", h5g.TYPE: "Named Type"}
LS_FORMAT = " %-10s    %-10s"

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
        return normpath(join(self.path,path))

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
        dname = dirname(path)
        bname = basename(path)
        try:
            if bname != '' and not self.file[dname].id.get_objinfo(bname).type == h5g.GROUP:
                self._error('"%s" is not an HDF5 group' % bname)
            else:
                self.path = path
        except:
            self._error('Can\'t open group "%s"' % path)

    def complete_cd(self, text, line, begidx, endidx):
        text = text.strip()
        grpname = self.abspath(dirname(text))
        targetname = basename(text)

        grp = self.file[grpname]
        rval = [join(grpname,x) for x in grp \
                    if x.find(targetname) == 0 and \
                    grp.id.get_objinfo(x).type == h5g.GROUP]
        return rval
    
    def do_ls(self, line):
        """ List contents of the specified group, or this one """

        LONG_STYLE = False
        try:
            opts, args = gnu_getopt(line.split(), 'l')
        except GetoptError, e:
            self._error(e.msg.capitalize())
            return

        if '-l' in [ opt[0] for opt in opts]:
            LONG_STYLE = True
        if len(args) == 0:
            grpname = self.path
        elif len(args) == 1:
            grpname = self.abspath(args[0])
        else:
            self._error("Too many arguments")
            return

        try:
            grp = self.file[grpname]
            if LONG_STYLE:
                print LS_FORMAT % ("Name", "Type")
                print LS_FORMAT % ("----", "----")
            for name in grp:
                typecode = grp.id.get_objinfo(name).type
                pname = name if typecode != h5g.GROUP else name+'/'
                if LONG_STYLE:
                    print LS_FORMAT % (pname, NAMES[typecode])
                else:
                    print pname
        except:
            self._error('Can\'t list contents of group "%s"' % hbasename(grpname))

    def complete_ls(self, *args):
        return self.complete_cd(*args)













