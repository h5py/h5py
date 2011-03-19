
import re

function_pattern = r'(?P<code>(unsigned[ ]+)?[a-zA-Z_]+[a-zA-Z0-9_]*\**)[ ]+(?P<fname>[a-zA-Z_]+[a-zA-Z0-9_]*)[ ]*\((?P<sig>[a-zA-Z0-9_,* ]*)\)'
sig_pattern = r'(?:[a-zA-Z_]+[a-zA-Z0-9_]*\**)[ ]+[ *]*(?P<param>[a-zA-Z_]+[a-zA-Z0-9_]*)'

fp = re.compile(function_pattern)
sp = re.compile(sig_pattern)

class BadLineError(Exception):
    pass

class UnknownCodeError(Exception):
    pass

class FunctionCruncher(object):

    def load_line(self, line):
        m = fp.match(line)
        if m is None:
            raise BadLineError("Line <<%s>> did not match regexp" % line)
        self.code = m.group('code')
        self.fname = m.group('fname')
        self.sig = m.group('sig')
        
        m = sp.findall(self.sig)
        if m is None:
            raise BadLineError("Signature for line <<%s>> did not match regexp" % line)
        self.sig_parts = m

        if '*' in self.code:
            self.condition = "==NULL"
            self.retval = "NULL"
        elif self.code in ('int', 'herr_t', 'htri_t', 'hid_t','hssize_t','ssize_t') \
          or re.match(r'H5[A-Z]+_[a-zA-Z_]+_t',self.code):
            self.condition = "<0"
            self.retval = "-1"
        elif self.code in ('unsigned int','haddr_t','hsize_t','size_t'):
            self.condition = "==0"
            self.retval = 0
        else:
            raise UnknownCodeError("Return code <<%s>> unknown" % self.code)

    def put_cython_signature(self):
        
        return "cdef %s %s_py(%s) except? %s" % (self.code, self.fname,
                                              self.sig, self.retval)

    def put_cython_wrapper(self):

        code_dict = {'code': self.code, 'fname': self.fname,
             'sig': self.sig, 'args': ", ".join(self.sig_parts),
             'condition': self.condition, 'retval': self.retval}

        code = """\
cdef %(code)s %(fname)s(%(sig)s) except? %(retval)s:
    cdef %(code)s r;
    r = c_%(fname)s(%(args)s)
    if r%(condition)s:
        if set_exception():
            return %(retval)s;
    return r
"""
        return code % code_dict

    def put_cython_import(self):

        return '%s c_%s "%s" (%s)' % (self.code, self.fname, self.fname, self.sig)

    def put_name(self):
        
        return self.fname

if __name__ == '__main__':

    fc = FunctionCruncher()
    f = open('auto_functions.txt','r')
    f_pxd = open('auto_defs.pxd','w')
    f_pyx = open('auto_defs.pyx','w')
    f_names = open('auto_names.txt','w')

    f_pxd.write("# This file is auto-generated.  Do not edit.\n\n")
    f_pyx.write("# This file is auto-generated.  Do not edit.\n\n")

    defs = "DEF H5PY_18API=1\n"
    defs += 'include "defs_types.pxi"\n'
    defs += 'cdef extern from "hdf5.h":\n'
    sigs = ""
    wrappers = ""
    names = ""

    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            fc.load_line(line)
        except BadLineError:
            print "skipped <<%s>>" % line
            continue
        defs += "  "+fc.put_cython_import()+"\n"
        sigs += fc.put_cython_signature()+"\n"
        wrappers += fc.put_cython_wrapper()+"\n"
        names += fc.put_name()+"\n"

    f_pxd.write(defs)
    f_pxd.write("\n\n")
    f_pxd.write(sigs)
    f_pyx.write(wrappers)
    f_names.write(names)

    f_pxd.close()
    f_pyx.close()
    f_names.close()



