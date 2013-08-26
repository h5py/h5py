#!/usr/bin/python
#-------------------------------------------------------------------------------
# Name:        c2py
# Purpose:     converts a C++ header to a python header.
#              This programm converts enums, structs and functions of hdf5.h to
#              api_functions.txt, which is used to build Cython code
#              the program can be called with: -h to see the argument list.
#              to compare the original list with the new generated list, following calls can help:
#
#  rm -f hdfFuncLst*.txt
#  ./c2py.py -v --hdfDir=../../hdf5/include --header=hdf5py.h hdfFuncLstB1.txt
#  grep -E -h '^\s+\w' api_functions.txt |sed -r -e 's/^\s+//' -e 's/\s+/ /g' -e 's/\s*\*\s*/ \*/g'| sort -k2 > hdfFuncLstA2.txt
#  grep -E -h '^\s+\w' hdfFuncLstB1.txt |sed -r -e 's/^\s+//' -e 's/\s+/ /g' -e 's/\s*\*\s*/ \*/g'| sort -k2 > hdfFuncLstB2.txt
#  meld hdfFuncLstA2.txt hdfFuncLstB2.txt&
#
# Author:      Thierry Zamofing
#
# Created:     15.05.2013
# Copyright:   (c) Thierry 2013
# Licence:     thierry.zamofing@psi.ch
#-------------------------------------------------------------------------------
'''
Parse the hdf5 header to use for h5py.

generates file: "outFile"
generates file: new.api_functions.txt,  unused.api_functions.txt
generates file: new.api_types_hdf5.pxd, unused.api_types_hdf5.pxd

This will generate the files new.api_functions.txt and new.api_types_hdf5.pxd that can
be compared with the origin h5py-files api_functions.txt and api_types_hdf5.pxd.
The generated "outFile" contains all declared functions, enums and structure
in a readable, condensed format.
the generated files unused.api_functions.txt and unused.api_types_hdf5.pxd contains functions, struct and enums,
that are not yet part of the h5py library.
'''
from __future__ import print_function, division
from pycparser import c_parser, c_ast, parse_file
import sys,os,re
import subprocess
import StringIO
import logging
_log=logging.getLogger(__name__)

class Chdr2py(c_ast.NodeVisitor):
  def __init__(self):
    self.out=''
    self.fn=''
    self.funcDict=dict()
    self.enumDict=dict()
    self.structDict=dict()
    
  @staticmethod
  def GetFuncList():
    cmd="grep -E '^\s+\w' h5py/api_functions.txt |grep -oE '\w+\('"
    #cmd="grep '^ [^(]*(' api_functions.txt | awk '{FS=\"[ (]*\"; print $3}'"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retval = p.wait()
    lst=[]
    for l in p.stdout.readlines():
      l=l[:-2]
      if l:
        lst.append(l)
    return lst
  
  def GenFuncListFile(self,fileName,unusedFileName):
    fl=Chdr2py.GetFuncList()
    flSet=set()
    fs=open(fileName,'w')
    for funcName in fl:
      fn=funcName
      #the h5py used the old API so H5Dopen is equal to H5Dopen1
      if funcName in ('H5Dopen','H5Gcreate','H5Gopen','H5Pget_filter','H5Pget_filter_by_id',
                      'H5Rget_obj_type','H5Topen','H5Tcommit','H5Tarray_create','H5Tget_array_dims',
                      'H5Acreate','H5Aiterate','H5Eclear','H5Eset_auto','H5Eget_auto','H5Eprint','H5Ewalk'):
        fn+='1'
      try:
        (retStr,paramLst)=v.funcDict[fn]
        flSet.add(fn)
      except KeyError as e:
        _log.warn('Function '+str(e)+' not found')
        continue
      s='  '+retStr+' '+' '*(9-len(retStr))+funcName+'('+', '.join(paramLst)+')\n'
      fs.write(s)
    
    fs.close()      
    
    if unusedFileName:
      flSet2=set(v.funcDict.keys())
      flSet2.difference_update(flSet)
      fs=open(unusedFileName,'w')
  
      fs.write('\n\n# ========= Missing functions =========\n')
      flSet2=list(flSet2);flSet2.sort()
      for funcName in flSet2:
        (retStr,paramLst)=v.funcDict[funcName]
        s='  '+retStr+' '+funcName+'('+', '.join(paramLst)+')\n'
        fs.write(s)
      fs.close()      
    print('generated file: %s %s'%(fileName,unusedFileName))

  @staticmethod
  def GetEnumStructList():
    cmd="grep -E 'def (enum|struct)' h5py/api_types_hdf5.pxd | awk '{FS=\"[ (:]*\"; print $4}'"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retval = p.wait()
    lst=[]
    for l in p.stdout.readlines():
      l=l[:-1]
      if l:
        lst.append(l)
    return lst

  def GenEnumStructFile(self,fileName,unusedFileName):
    esl=Chdr2py.GetEnumStructList()
    eSet=set()
    sSet=set()
    #the h5py used the old API so H5Dopen is equal to H5Dopen1       
    typeNameTranslate={'H5FD_mem_t':'H5F_mem_t', #typedef enum H5F_mem_t  H5FD_mem_t;
                       }
    typeIgnore=set(('H5G_link_t', # defines in H5Gpublic.h:
                                  #define H5G_SAME_LOC H5L_SAME_LOC
                                  #define H5G_LINK_ERROR H5L_TYPE_ERROR
                                  #define H5G_LINK_HARD H5L_TYPE_HARD
                                  #define H5G_LINK_SOFT H5L_TYPE_SOFT
                                  #define H5G_link_t H5L_type_t
                    'space', 'mesg', # are structs in type H5O_hdr_info_t
                    'meta_size', # is a struct in type H5O_info_t
                    ))
    fs=open(fileName,'w')
    for typeName in esl:
      if typeName in typeIgnore:
        continue
      tn=typeNameTranslate.get(typeName,typeName)
      if self.enumDict.has_key(tn):
        eSet.add(tn)
        enum=self.enumDict[tn]
        s='  ctypedef enum '+typeName+':\n'

        tab=0
        for e in enum:
          tab=max(tab,4+len(e[0]))
        for e in enum:
          s+='    '+e[0]
          if e[1]!=None:
            s+=' '*(tab-2-len(e[0]))
            s+='=% d'%e[1]
          s+='\n'
        s=s[:-1]+'\n\n'
      elif self.structDict.has_key(typeName):
        sSet.add(typeName)
        struct=self.structDict[typeName]
        s='  ctypedef struct '+typeName+':\n'
        tab=0
        for t in struct:
          tab=max(tab,4+len(t[0]))
        for t in struct:
          s+='    '+t[0]
          s+=' '*(tab-2-len(t[0]))
          s+=''.join(t[1:])+'\n'
        s=s[:-1]+'\n\n'
      else:
        _log.warn('Key Error struct or enum '+typeName+' not found')
        continue

      fs.write(s)
    fs.close()      
    if unusedFileName:
      fs=open(unusedFileName,'w')
      eSet2=set(v.enumDict.keys())
      eSet2.difference_update(eSet)
  
      fs.write('\n\n# ========= Missing enums =========\n')
      eSet2=list(eSet2);eSet2.sort()
      for typeName in eSet2:
        enum=self.enumDict[typeName]
        s='  ctypedef enum '+typeName+':\n'
        fs.write(s)

      sSet2=set(v.structDict.keys())
      sSet2.difference_update(sSet)
      fs.write('\n\n# ========= Missing structs =========\n')
      sSet2=list(sSet2);sSet2.sort()
      for typeName in sSet2:
        struct=self.structDict[typeName]
        s='  ctypedef struct '+typeName+':\n'
        fs.write(s)

      
      
      fs.close()      
    print('generated file: %s %s'%(fileName,unusedFileName))
    
  @staticmethod
  def type2str(node):
    #unsigned int **  var [][3]
    #t0           t1  t2  t3
    t=[[],'','','']
    nd=node
    while True:
      if type(nd)==c_ast.Decl:
        pass
      elif type(nd)==c_ast.Typename:
        pass
      elif type(nd)==c_ast.PtrDecl:
        t[1]+='*'
      elif type(nd)==c_ast.ArrayDecl:
        if type(nd.dim)==c_ast.Constant:
          v=nd.dim.value
        else:
          v=''
        t[3]+='['+v+']'
      elif type(nd)==c_ast.TypeDecl:
        if nd.quals:
          t[0]+=nd.quals
        if nd.declname:
          t[2]+=nd.declname
        else:
          pass
      else:
        break  
      nd=nd.type  
    if type(nd)==c_ast.IdentifierType:
      t[0]+=nd.names
    elif type(nd)==c_ast.EllipsisParam:
      t[2]+='...'
    else:
      s = StringIO.StringIO();
      if logging.root.level<=logging.DEBUG:
        s.write('\n')
        nd.show(s)
      _log.warn(node.coord.file+':'+str(node.coord.line)+':can''t handle type'+str(t)+s.getvalue())
      t[0].append('???')

    t[0]=' '.join(t[0])
    return (t,nd)
    
  @staticmethod
  def enumval2str(node):
    if type(node)==c_ast.UnaryOp:
      expr=node.op+Chdr2py.enumval2str(node.expr)
    elif type(node)==c_ast.BinaryOp:
      expr=Chdr2py.enumval2str(node.left)+node.op+Chdr2py.enumval2str(node.right)
    elif type(node)==c_ast.Constant:
      expr=node.value
    else:
      raise BaseException('unsupported')
    return expr

  def visit_FuncDecl(self, node):
    nd=node.type

    (t,nd)=Chdr2py.type2str(node.type)
    
    if t[0] and t[2]:
      funcName=t[2]
      if funcName=='H5Eget_major':
        pass
      retStr=t[0]+t[1]
    else:
      s = StringIO.StringIO();node.show(s)
      _log.warn(node.coord.file+':'+str(node.coord.line)+':can''t handle type'+str(t)+'\n'+s.getvalue())
      return
    _log.debug(node.coord.file+':'+str(node.coord.line)+':'+funcName)
    if self.fn!=node.coord.file:
      if not self.fn:
        self.out+='hdf5:\n'
      if node.coord.file.endswith('H5DOpublic.h'):
        self.out+='hdf5_hl:\n'
      self.fn=node.coord.file
      self.out+='#'+self.fn+'\n'
    paramLst=[]
    for p in node.args.params:
      #unsigned int **  var [][3]
      #t0           t1  t2  t3     
      (t,nd)=Chdr2py.type2str(p)
      if t[2]=='type':
        t[2]+='_id'
      if t[3]=='[]':
        t[1]+='*'; t[3]=''
      if t!=['void','','','']:
        t[1]=' '+t[1]
        paramLst.append(''.join(t))
           
    s='  '+retStr+' '+funcName+'('+', '.join(paramLst)+')\n'
    self.out+=s
    self.funcDict[funcName]=(retStr,paramLst)
    pass

  def visit_TypeDecl(self, node):
    try:
      typeName=node.declname
      t=type(node.type)
      _log.debug(node.coord.file+':'+str(node.coord.line)+':'+t.__name__+':'+typeName)
    except AttributeError as e:
      s = StringIO.StringIO()
      if logging.root.level<=logging.DEBUG:
        s.write('\n')
        node.show(s)
      _log.warn('invalid TypeDecl node'+s.getvalue())
      
    if t==c_ast.Enum:
      eVal=None#eVal=0
      enum=[]
      if not node.type.values:
        s = StringIO.StringIO()
        if logging.root.level<=logging.DEBUG:
          s.write('\n')
          node.show(s)
        _log.warn(node.coord.file+':'+str(node.coord.line)+':empty enum ignored'+s.getvalue())
        return
      for e in node.type.values.enumerators:
        eVal=e.value
        if eVal:
          try:
            res=Chdr2py.enumval2str(eVal)
            if res!=None:
              eVal=eval(res)
          except TypeError:
            #eVal=0
            s = StringIO.StringIO();e.show(s)
            _log.warn(node.coord.file+':'+str(node.coord.line)+'can''t get value of:'+s.getvalue())
        enum.append((e.name,eVal))
        #eVal+=1
      s='  ctypedef enum '+typeName+':\n'
      for e in enum:
        s+='    '+e[0]
        if e[1]:
          s+='='+str(e[1])
        s+=',\n'
      s=s[:-2]+'\n\n'
      self.out+=s
      self.enumDict[typeName]=enum
 
    elif t==c_ast.Struct:
      struct=[]
      if not node.type.decls: #ignore empty types
        s = StringIO.StringIO()
        if logging.root.level<=logging.DEBUG:
          s.write('\n')
          node.show(s)
        _log.warn(node.coord.file+':'+str(node.coord.line)+':empty struct ignored'+s.getvalue())
        return
      for d in node.type.decls:
        (t,nd)=Chdr2py.type2str(d)
        struct.append(t)

      outStr='  ctypedef struct '+typeName+':\n'

      for t in struct:
        t[1]=' '+t[1]
        outStr+='    '+''.join(t)+',\n'
      outStr=outStr[:-2]+'\n\n'
      self.structDict[typeName]=struct
    else:
      pass

def CheckDuplicate(args):
  '''
  seeks structs and enums that are declared twice
  '''
  _log.info('')
  #for fn in os.listdir('.'):
  #  if
  cmd="grep -nE 'c\w+\s+(enum|struct)\s+\w+\s*:' *.p* --exclude=new.* --exclude=unused.*"
  p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  retval = p.wait()
  dictType=dict()
  for l in p.stdout.readlines():
    m=re.match('(.*):(.*):\s*(\w+)\s*(\w+)\s*(\w+)\s*:', l[:-1])
    if m:
      g=m.groups()
      #print(g)
      ll=dictType.get(g[-1])
      if ll:
        ll.append(g[:-1])
        pass
      else:
        dictType[g[-1]]=[g[:-1]]
    else:
      _log.warn('Regexp Failed:'+l[:-1])

  for (k,v) in dictType.iteritems():
    if len(v)>1:
      print(k)
      for vv in v:
        print('  ',vv)
  pass

def CheckUsage(args):
  '''
  seeks structs and enums that are declared twice
  '''
  _log.info('')
  funcList=Chdr2py.GetFuncList()

  lstFuncUse=list()
  for func in funcList:   
    cmd="grep -nE '\W"+func+"\W' *.p* --exclude=defs.pxd --exclude=defs.pyx --exclude=_hdf5.pxd --exclude=c2py.py"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retval = p.wait()
    lstUse=list()
    for l in p.stdout.readlines():
      m=re.match('(.*):(.*):\s*(.*)', l[:-1])
      if m:
        g=m.groups()
        #print(g)
        lstUse.append(g)
      else:
        _log.warn('Regexp Failed:'+l[:-1])
    lstFuncUse.append([func,lstUse])
    if not lstUse:
      _log.warn('Unused function:'+func)
    if not lstUse:
      print('-'+func+' unused function' )
    else:
      print(func)
      for v in lstUse:
        print('  ',v)
     
    pass
  pass


  return lstFuncUse

if __name__ == '__main__':
  """this module parse the libHeLIC header file and generates the python wrapper
     the used pycparser Version is: '2.0.6'. check with
       import pycparser as pp;print pp.__version__
     further the MingW32 gcc compiler must be installed.
  """
  import argparse

  hdlr=logging.StreamHandler()
  fmt = logging.Formatter('%(levelname)s:%(name)s:%(lineno)d:%(funcName)s:%(message)s')
  hdlr.setFormatter(fmt)
  del logging.root.handlers[:]
  logging.root.addHandler(hdlr)
  logging.root.setLevel(logging.INFO)
  
  cpp_args= [r'-D_PYPARSE_']
  cmdLst=['gen', 'dup', 'use']
  exampleCmd=r'-v --hdfDir=../hdf5/include --header=hdf5py.h --outFile=c2pyOut.txt gen'
  parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                   description=__doc__,
                                   epilog='Example:\n  '+os.path.basename(sys.argv[0])+' '+exampleCmd+'\n ')
  parser.add_argument('command', choices=cmdLst, default=cmdLst[0], help='the command to execute')
  parser.add_argument('--outFile', help='the file to generate')
  parser.add_argument('--hdfDir', help='the hdf directory')
  parser.add_argument('--header', help='the header file to parse')
  parser.add_argument('-v', action='store_true', help='verbose')

  #args = parser.parse_args(.split())
  if len(sys.argv)==1:
    #print('running with arguments: '+exampleCmd)
    #args = parser.parse_args(exampleCmd.split())
    args = parser.parse_args(['-h',])
  else:
    args = parser.parse_args()

  if args.command==cmdLst[0]:#gen
    fnHdr=args.header
    fnOut=args.outFile
    cpp_args.append(r'-I'+args.hdfDir)
    
    if args.v:
      logging.root.setLevel(logging.DEBUG)
  
    print(fnOut,fnHdr,cpp_args) 
  
    ast = parse_file(fnHdr, use_cpp=True, cpp_args=cpp_args)
    v = Chdr2py()
    #ast.show()
    v.visit(ast)
  
    if fnOut:
      fs=open(fnOut,'w')
      fs.write('#-------------------------------- GENERATED c2py.py ----------------------------\n')
      fs.write(v.out)
      fs.write('#-------------------------------- GENERATED END --------------------------------\n')
      fs.close()
      print('generated file:'+fnOut)
  
    v.GenFuncListFile('h5py/new.api_functions.txt','h5py/unused.api_functions.txt')
    v.GenEnumStructFile('h5py/new.api_types_hdf5.pxd','h5py/unused.api_types_hdf5.pxd')
  elif args.command==cmdLst[1]:#dup
    CheckDuplicate(args)
  elif args.command==cmdLst[2]:#use
    CheckUsage(args)

  
  pass

