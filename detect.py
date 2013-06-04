from distutils import ccompiler
import os, subprocess

def detect(**s):

    outfiles = []
    
    cc = ccompiler.new_compiler()
    try:
        outfiles = cc.compile(['detect.c'], include_dirs = s.get('include_dirs'),
            macros = s.get('define_macros'))
        cc.link_executable(outfiles, 'detect', libraries = s.get('libraries'),
            library_dirs = s.get('library_dirs'),
            runtime_library_dirs = s.get('runtime_library_dirs'))
        outfiles += ['detect']
        output = subprocess.check_output(['./detect']).decode('ascii')
        version = tuple(int(x) for x in output.split('.'))
    except Exception as e:
        print("Failed to detect HDF5 version; defaulting to 1.8.4")
        print(e)
        return (1, 8, 4)
    finally:
        for fname in outfiles:
            os.unlink(fname)
    return version

