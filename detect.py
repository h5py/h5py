
import os.path as op

def detect_hdf5(basedir, **compiler_attrs):
    """ Compile, link & execute a test program, in empty directory basedir.
    The C compiler will be updated with any keywords given via setattr.

    Returns a dictionary containing information about the HDF5 installation.
    """

    from distutils import ccompiler
    from distutils.core import CompileError, LinkError
    import subprocess

    cc = ccompiler.new_compiler()
    for name, val in compiler_attrs.iteritems():
        setattr(cc, name, val)

    cfile = op.join(basedir, 'vers.c')
    efile = op.join(basedir, 'vers')

    f = open(cfile, 'w')
    try:
        f.write(
r"""
#include <stdio.h>
#include "hdf5.h"

int main(){
    unsigned int main, minor, release;
    if(H5get_libversion(&main, &minor, &release)<0) return 1;
    fprintf(stdout, "vers: %d.%d.%d\n", main, minor, release);
    return 0;
}
""")
    finally:
        f.close()

    objs = cc.compile([cfile])
    cc.link_executable(objs, efile)

    result = subprocess.Popen(efile,
             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    so, se = result.communicate()
    if result.returncode:
        raise IOError("Error running HDF5 version detection script:\n%s\n%s" % (so,se))

    handlers = {'vers':     lambda val: tuple(int(v) for v in val.split('.')),
                'parallel': lambda val: bool(int(val))}

    props = {}
    for line in (x for x in so.split('\n') if x):
        key, val = line.split(':')
        props[key] = handlers[key](val)

    return props

