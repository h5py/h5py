from __future__ import with_statement

"""
    Script to test multiple configurations of h5py on a target machine
"""

import os.path as op
import os, sys
from commands import getstatusoutput

def debug(what):
    if 'debug' in sys.argv:
        print '>>> '+what

def iterconfigs(cfile):
    """ Iterate over multiple configurations; i.e. the "whatever" in 
        in "setup.py configure whatever".  The special value DEFAULT
        corresponds to no extra config information.
    """
    for line in (x.strip() for x in cfile):
        if len(line) > 0 and not line.startswith('#'):
            debug("Line: "+line)
            if line == 'DEFAULT':
                yield ""
            else:
                yield line

class CommandFailed(Exception):
    pass

def do_cmd(cmd):
    debug(cmd)
    s, o = getstatusoutput(cmd)
    if s != 0:
        msg = "Command failed: %s" % cmd
        msg += '\n'+'-'*len(msg)
        print msg
        raise CommandFailed(cmd)
    
def run():

    failed = False

    # Check what versions of Python are installed
    pythons = [x for x in ('python2.5', 'python2.6') if os.system('%s -V > /dev/null 2>&1' % x) == 0]

    debug("Have pythons %s" % pythons)

    # Try to open configs file
    try:
        with open(op.join(op.expanduser('~'), '.h5pytest'),'r') as cfile:
            debug("Config file found!")
            configs = list(iterconfigs(cfile))
    except IOError:
        debug("No config file")
        configs = [""]

    for p in pythons:
        for c in configs:
            try:
                do_cmd('%s setup.py configure %s' % (p, c))
                do_cmd('%s setup.py build' % p)
                do_cmd('%s setup.py test' %p)
            except CommandFailed:
                failed = True
            finally:
                do_cmd('%s setup.py clean' %p)

    return not failed

if __name__ == '__main__':
    if not run():
        sys.exit(1)


