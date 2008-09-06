# Development script to test setup options
import os.path as op
import os, sys

# HDF5 library versions
libnames = ['h167','h180','h181']
lib_base = op.abspath(op.join(op.curdir, '..'))
libs = [''] + [' --hdf5='+op.join(lib_base, x) for x in libnames]

# Experimental non-blocking I/O
nonblock = ['']

# API versions
api = [' --api=16', ' --api=18']

linebase = "python setup.py test --pyrex-force%s%s%s"
print libs
print nonblock
print api

idx=0
for l in libs:
    for n in nonblock:
        for a in api:
                # Only allow --api=18 if 1.8.X library being tested
                if '8' in a and not '8' in l:
                    continue
                line = linebase % (l, n, a)
                outfile = "autotest%d.txt" % idx
                print 'Testing config %d "%s"...' % (idx, line)
                retval = os.system(line+" > %s 2>&1" % outfile)
                if retval !=0:
                    print '!!! Line failed; output saved to "%s"' % outfile
                else:
                    os.unlink(outfile)
                idx += 1
