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

"""
    Internal module which provides the guts of the File.browse() method
"""

from cmd import Cmd
from posixpath import join, basename, dirname, normpath, isabs
from getopt import gnu_getopt, GetoptError
import shlex
import os
import re
import sys

from utils_hl import hbasename

from h5py import h5g

NAMES = {h5g.DATASET: "Dataset", h5g.GROUP: "Group", h5g.TYPE: "Named Type"}
LS_FORMAT = " %-20s    %-10s"

class CmdError(StandardError):
    pass

# Why the hell doesn't Cmd inherit from object?  Properties don't work!
class _H5Browser(Cmd, object):

    """
        HDF5 file browser class which holds state between sessions.
    """
    def _setpath(self, path):
        self.prompt = "HDF5: %s> " % (hbasename(path))
        self._path = path

    path = property(lambda self: self._path, _setpath)

    def __init__(self, fileobj, path=None, importdict=None):
        """ Browse the file, putting any imported names into importdict. """
        Cmd.__init__(self)
        self.file = fileobj

        self.path = path if path is not None else '/'

        self.importdict = importdict
        self.cmdloop('Browsing "%s". Type "help" for commands, "exit" to exit.' % os.path.basename(self.file.name))

    def onecmd(self, line):
        retval = False
        try:
            retval = Cmd.onecmd(self, line)
        except (CmdError, GetoptError), e:
            print "Error: "+e.args[0]
        return retval

    def abspath(self, path):
        """ Correctly interpret the given path fragment, relative to the
            current path.
        """
        return normpath(join(self.path,path))

    def do_exit(self, line):
        """ Exit back to Python """
        return True

    def do_EOF(self, line):
        """ (Ctrl-D) Exit back to Python """
        return True

    def do_pwd(self, line):
        """ Print name of current group """
        print self.path

    def do_cd(self, line):
        """ cd [group] """
        args = shlex.split(line)
        if len(args) > 1:
            raise CmdError("Too many arguments")
        path = args[0] if len(args) == 1 else ''

        path = self.abspath(path)
        dname = dirname(path)
        bname = basename(path)
        try:
            if bname != '' and not self.file[dname].id.get_objinfo(bname).type == h5g.GROUP:
                raise CmdError('"%s" is not an HDF5 group' % bname)
            else:
                self.path = path
        except:
            raise CmdError('Can\'t open group "%s"' % path)

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
        """ ls [-l] [group] """

        LONG_STYLE = False
        opts, args = gnu_getopt(shlex.split(line), 'l')

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
                print 'Group "%s" in file "%s":' % (hbasename(grpname), os.path.basename(self.file.name))
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
            raise CmdError('Can\'t list contents of group "%s"' % hbasename(grpname))
        
    def do_info(self, line):

        opts, args = gnu_getopt(shlex.split(line),'')

        for arg in args:
            name = self.abspath(arg)
            try:
                obj = self.file[name]
                print obj.desc()
            except:
                raise CmdError("Can't get info on object \"%s\"" % hbasename(name))

    def complete_info(self, text, line, begidx, endidx):
        text = text.strip()
        grpname = self.abspath(dirname(text))
        targetname = basename(text)

        grp = self.file[grpname]
        rval = [join(grpname,x) for x in grp \
                    if x.find(targetname) == 0]
        return rval


    def do_import(self, line):
        """ import name [as python_name] 
 import name1 name2 name3 name4 ...
        """
        if self.importdict is None:
            raise CmdError("No import dictionary provided")

        opts, args = gnu_getopt(shlex.split(line),'')
        
        pynames = []
        hnames = []

        importdict = {}   # [Python name] => HDF5 object

        if len(args) == 3 and args[1] == 'as':
            pynames.append(args[2])
            hnames.append(args[0])
        else:
            for arg in args:
                absname = self.abspath(arg)
                pynames.append(basename(absname))
                hnames.append(absname)

        for pyname, hname in zip(pynames, hnames):
            try:
                obj = self.file[hname]
            except Exception, e:
                raise CmdError("Can't import %s" % pyname)

            if len(re.sub('[A-Za-z_][A-Za-z0-9_]*','',pyname)) != 0:
                raise CmdError("%s is not a valid Python identifier" % pyname)

            if pyname in self.importdict:
                if not raw_input("Name %s already in use. Really import (y/N)?  " % pyname).strip().lower().startswith('y'):
                    continue

            importdict[pyname] = obj

        self.importdict.update(importdict)

    def complete_import(self, text, line, begidx, endidx):
        text = text.strip()
        grpname = self.abspath(dirname(text))
        targetname = basename(text)

        grp = self.file[grpname]
        rval = [join(grpname,x) for x in grp \
                    if x.find(targetname) == 0]
        return rval


    def complete_ls(self, *args):
        return self.complete_cd(*args)




