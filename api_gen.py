#!/usr/bin/python
#-------------------------------------------------------------------------------
# Name:        api_gen.py [--stub] [-v]
# Purpose:     generates files from api_functions.txt that are used
#              used to talk to the HDF5 api
#              The following files are used to talk to the HDF5 api:
#               - hdf5_types.pxd:   HDF5 type definitions       (static)
#               - _hdf5.pxd:        HDF5 function signatures    (generated with that script)
#               - defs.pxd:         HDF5 function proxy defs    (generated with that script)
#               - defs.pyx:         HDF5 function proxies       (generated with that script)
#
#              Optional arguments are:
#              --stub to create stub functions without exception handling
#              -v     for verbose
#
# Author:      ...
# Created:     ...
# Copyright:   ...
# Licence:     ...
#-------------------------------------------------------------------------------

#

import re
import warnings
import os.path as op

class BadLineError(Exception):
  pass

class UnknownCodeError(Exception):
  pass

regexFunc  = re.compile('\s*(?P<ret>.*?)\s*(?P<func>\w*)\s*\((?P<sig>[\w\s,*\[\]]*)\)')
regexParam = re.compile('(?P<param>\w+)\s*(?:\[|,|$)')

class strTbl:
  preambleRaw="""\
from api_types_hdf5 cimport *
from api_types_ext cimport *

"""
  preambleDef="""\
from api_types_hdf5 cimport *
from api_types_ext cimport *

"""
  preambleImp="""\
from api_types_ext cimport *
from api_types_hdf5 cimport *

cimport _hdf5

from _errors cimport set_exception

include "_locks.pxi"

rlock = FastRLock()
"""

  tplRaw="  %(ret)s %(func)s(%(sig)s) except *\n"
  tplDef="cdef %(ret)s %(func)s(%(sig)s) except *\n"

  tplImp = """\
cdef %(ret)s %(func)s(%(sig)s) except *:
  cdef %(ret)s r
  with rlock:
    r = _hdf5.%(func)s(%(args)s)
    if r%(condition)s:
      set_exception()
    return r

"""

  tplImpStub = """\
cdef %(ret)s %(func)s(%(sig)s) except *:
  with rlock:
    return hdf5.%(func)s(%(args)s)

"""


class FunctionCruncher(object):

  def __init__(self, stub, verbose):
    self.stub = stub
    self.verbose = verbose
    self.retTypes=set()

  def run(self):
    self.strRaw=''
    self.strDef=''
    self.epiDef='' #epilog in defs.pxd
    self.strImp=''


    # Function definitions file
    fsAPIFunc = open(op.join('h5py', 'api_functions.txt'),'r')
    for line in fsAPIFunc:
      if not line or line[0] == '#' or line[0] == '\n':
        continue
      try:
        self.handle_line(line)
      except BadLineError:
        warnings.warn("Skipped <<%s>>" % line)

    fsAPIFunc.close()

    # Create output files
    filenames=('_hdf5.pxd','defs.pxd','defs.pyx')

    fsRaw = open(op.join('h5py', filenames[0]),'w')
    fsRaw.write(strTbl.preambleRaw)
    fsRaw.write(self.strRaw)
    fsRaw.close()

    fsDef = open(op.join('h5py', filenames[1]),'w')
    fsDef.write(strTbl.preambleDef)
    fsDef.write(self.strDef)
    self.epiDef+='  pass\n'#needed because there are no "hdf5_hl.h" in this block
    fsDef.write(self.epiDef)
    fsDef.close()

    fsImp = open(op.join('h5py', filenames[2]),'w')
    fsImp.write(strTbl.preambleImp)
    fsImp.write(self.strImp)
    fsImp.close()

    if self.verbose:
      print('Existing ReturnTypes are:')
      rtLst=list(self.retTypes)
      rtLst.sort()
      for rt in rtLst:
        print('  '+rt)

    print('files:'+', '.join(filenames)+' generated.')

  def handle_line(self, line):
    """ Parse a function definition line and output the correct code
    to each of the output files. """

    if line.startswith(' '):
      line = line.strip()
      if line.startswith('#'):
        return
      m = regexFunc.match(line)
      if m is None:
        raise BadLineError(
            "Signature for line <<%s>> did not match regexp" % line
            )
      dictFuncElem = m.groupdict()
      
      dictFuncElem['sig']=dictFuncElem['sig'].replace('const ', 'const_')
      dictFuncElem['sig']=dictFuncElem['sig'].replace('const_unsigned char', 'const_unsigned_char')
      dictFuncElem['sig']=dictFuncElem['sig'].replace('const_unsigned int', 'const_unsigned_int')
      dictFuncElem['sig']=dictFuncElem['sig'].replace('const_unsigned short', 'const_unsigned_short')
      dictFuncElem['sig']=dictFuncElem['sig'].replace('const_unsigned long', 'const_unsigned_long')
      dictFuncElem['sig']=dictFuncElem['sig'].replace('const_long long', 'const_long_long')

      if self.verbose:
        print dictFuncElem
      args = regexParam.findall(dictFuncElem['sig'])
      if args is None:
        raise BadLineError("Can't understand function signature <<%s>>" % dictFuncElem['sig'])
      args = ", ".join(args)
      dictFuncElem['args']=args

      # Figure out what conditional to use for the error testing
      ret = dictFuncElem['ret']
      self.retTypes.add(ret)
      if ret in ('H5T_conv_t',):
        dictFuncElem['condition']="==NULL"
        self.strRaw+=strTbl.tplRaw%dictFuncElem
        self.strDef+=strTbl.tplDef%dictFuncElem
        self.strImp+=(strTbl.tplImpStub if self.stub else strTbl.tplImp)%dictFuncElem
      elif ret in ('hsize_t','size_t'):
        dictFuncElem['condition']="==0"
        self.strRaw+=strTbl.tplRaw%dictFuncElem
        self.strDef+=strTbl.tplDef%dictFuncElem
        self.strImp+=(strTbl.tplImpStub if self.stub else strTbl.tplImp)%dictFuncElem
      elif ret in ('char*',): #no exception handling. only direct import
        self.epiDef+=strTbl.tplRaw%dictFuncElem
      elif ret in ('int', 'herr_t', 'htri_t', 'hid_t','hssize_t','ssize_t','haddr_t','H5S_sel_type') \
               or re.match(r'H5[A-Z]+_[a-zA-Z_]+_t$',ret):
        dictFuncElem['condition']="<0"
        self.strRaw+=strTbl.tplRaw%dictFuncElem
        self.strDef+=strTbl.tplDef%dictFuncElem
        self.strImp+=(strTbl.tplImpStub if self.stub else strTbl.tplImp)%dictFuncElem
      #elif '*' in ret or ret in ('H5T_conv_t',):
      #  condition = "==NULL"
      else:
        raise UnknownCodeError("return type <<%s>> unknown" % ret)
    else:
      inc = line.split(':')[0]
      self.strRaw+='cdef extern from "%s.h":\n' % inc
      self.epiDef+='\ncdef extern from "%s.h":\n' % inc

def run(stub=False):
    fc = FunctionCruncher(stub,False)
    fc.run()

if __name__ == '__main__':

  import sys
  class Args:
    pass
  args=Args()
  if '--stub' in sys.argv:
    args.stub = True
    print 'Stub functions (without exception handling) are generated'
  else:
    args.stub = False
  args.verbose = True if '-v' in sys.argv else False
  fc = FunctionCruncher(args.stub,args.verbose)
  fc.run()



