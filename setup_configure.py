from distutils.cmd import Command
import os
import pickle

def loadpickle():
    try:
        with open('xconfig.pkl','r') as f:
            cfg = pickle.load(f)
        if not isinstance(cfg, dict): raise TypeError
    except Exception:
        return {}
    return cfg
    
def savepickle(dct):
    try:
        with open('xconfig.pkl','w') as f:
            pickle.dump(dct, f)
    except Exception:
        pass
    
def validate_version(s):
    try:
        tpl = tuple(int(x) for x in s.split('.'))
        if len(tpl) != 3: raise ValueError
    except Exception:
        raise ValueError("HDF5 version string must be in X.Y.Z format")

class EnvironmentOptions(object):

    def __init__(self):
        self.hdf5 = os.environ.get('HDF5_DIR')
        self.hdf5_version = os.environ.get('HDF5_VERSION')
        if self.hdf5_version is not None:
            validate_version(self.hdf5_version)
            
class configure(Command):

    """ Determine config settings from the environment """

    description = "Run the test suite"

    user_options = [('hdf5=', 'h', 'Custom path to HDF5'),
                    ('hdf5-version=', '5', 'HDF5 version "X.Y.Z"'),
                    ('mpi', 'm', 'Enable MPI building'),
                    ('reset', 'r', 'Reset config options') ]
    
    def initialize_options(self):
        self.hdf5 = None
        self.hdf5_version = None
        self.mpi = None
        self.reset = None
        
    def finalize_options(self):
        if self.hdf5_version is not None:
            validate_version(self.hdf5_version)

    def reset_rebuild(self):
        dct = loadpickle()
        dct['rebuild'] = False
        savepickle(dct)
        
    def run(self):
        env = EnvironmentOptions()
        
        oldsettings = {} if self.reset else loadpickle()
        dct = oldsettings.copy()
        
        if self.hdf5 is not None:
            dct['cmd_hdf5'] = self.hdf5
        if env.hdf5 is not None:
            dct['env_hdf5'] = env.hdf5
        if self.hdf5_version is not None:
            dct['cmd_hdf5_version'] = self.hdf5_version
        if env.hdf5_version is not None:
            dct['env_hdf5_version'] = env.hdf5_version
        if self.mpi is not None:
            dct['cmd_mpi'] = self.mpi

        self.rebuild_required = dct.get('rebuild') or dct != oldsettings
        if self.reset and not all((not x) for x in loadpickle().values()):
            self.rebuild_required = True
        dct['rebuild'] = self.rebuild_required
        
        savepickle(dct)
    
        # Now use precedence order to set the attributes
        if self.hdf5 is None:
            self.hdf5 = oldsettings.get('cmd_hdf5')
        if self.hdf5 is None:
            self.hdf5 = env.hdf5
        if self.hdf5 is None:
            self.hdf5 = oldsettings.get('env_hdf5')
            
        if self.hdf5_version is None:
            self.hdf5_version = oldsettings.get('cmd_hdf5_version')
        if self.hdf5_version is None:
            self.hdf5_version = env.hdf5_version
        if self.hdf5_version is None:
            self.hdf5_version = oldsettings.get('env_hdf5_version')
        if self.hdf5_version is None:
            self.hdf5_version = autodetect_version(self.hdf5)
            
        if self.mpi is None:
            self.mpi = oldsettings.get('cmd_mpi')
                
        print('*' * 80)
        print(' ' * 23 + "Summary of the h5py configuration")
        print('')
        print("    Path to HDF5: " + repr(self.hdf5))
        print("    HDF5 Version: " + repr(self.hdf5_version))
        print("     MPI Enabled: " + repr(bool(self.mpi)))
        print("Rebuild Required: " + repr(bool(self.rebuild_required)))
        print('')
        print('*' * 80)
        
def autodetect_version(libdir=None):
    """
    Detect the current version of HDF5, and return it as a tuple
    (major, minor, release).

    If the version can't be determined, prints error information to stderr
    and returns None.

    libdir: optional place to search for libhdf5.so.
    """

    import os
    import sys
    import os.path as op
    import re
    import ctypes
    from ctypes import byref

    if sys.platform.startswith('win'):
        regexp = re.compile('^hdf5.dll$')
    elif sys.platform.startswith('darwin'):
        regexp = re.compile(r'^libhdf5.dylib')
    else:
        regexp = re.compile(r'^libhdf5.so')

    try:
        path = None

        libdirs = [] if libdir is None else [libdir]
        libdirs += ['/usr/local/lib', '/opt/local/lib']
        for d in libdirs:
            try:
                candidates = [x for x in os.listdir(d) if regexp.match(x)]
                if len(candidates) != 0:
                    candidates.sort(key=lambda x: len(x))   # Prefer libfoo.so to libfoo.so.X.Y.Z
                    path = op.abspath(op.join(d, candidates[0]))
            except Exception:
                pass   # We skip invalid entries, because that's what the C compiler does

        if path is None:
            path = "libhdf5.so"

        lib = ctypes.cdll.LoadLibrary(path)

        major = ctypes.c_uint()
        minor = ctypes.c_uint()
        release = ctypes.c_uint()

        lib.H5get_libversion(byref(major), byref(minor), byref(release))

        vers = (int(major.value), int(minor.value), int(release.value))
        vers = "%s.%s.%s" % vers
        print("Autodetected HDF5 version " + vers)

        return vers

    except Exception as e:

        print(e)
        return '1.8.4'
