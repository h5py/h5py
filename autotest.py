# Development script to test setup options
import os.path as op
import os, sys

# Versions of the Python interpreter to test
python_versions = ['python2.5', 'python2.6']

# Expect these to exist in parent dir
libnames = ['h166',  'h180', 'h182']

# Additional options for each library version
extraopts = {'h182': ['--api=16']}

def runcmd(cmd, logfile=None):
    """ Execute a command, capturing output in a logfile.

    Logfile is deleted if command succeeds.  Aborts Python if
    exit code is exactly 1, else returns that value.
    """

    print "Executing %s" % cmd
    retval = os.system('%s > %s 2>&1' % (cmd, logfile if logfile is not None else '/dev/null'))
    if retval in (1,2):
        print "Exiting with internal exception"
        sys.exit(1)
    elif retval != 0:
        print '!!! Command "%s" failed with status %s; output saved to %s' % (cmd, retval, logfile)
    elif logfile is not None:
        os.unlink(logfile)

    return retval

retvals = []

for p in python_versions:
    for l in libnames:
        opts = ['']+extraopts.get(l,[])

        for i, o in enumerate(opts):

            outfile = 'autotest-%s-%s-%s.txt' % (l, i, p)

            retvals.append( runcmd('%s setup.py configure --hdf5=../%s %s' % (p, l, o)) )
            retvals.append( runcmd('%s setup.py build' % p, outfile) )
            retvals.append( runcmd('%s setup.py test' % p) )
            retvals.append( runcmd('%s setup.py clean' % p) )

            if not any(retvals):
                retvals = []
            else:
                sys.exit(17)
print 'Done'


