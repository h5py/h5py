
"""
    Implements a new custom Distutils command for handling library
    configuration.
    
    The "configure" command here doesn't directly affect things like
    config.pxi; rather, it exists to provide a set of attributes that are
    used by the build_ext replacement in setup_build.py.
    
    Options from the command line and environment variables are stored
    between invocations in a pickle file.  This allows configuring the library
    once and e.g. calling "build" and "test" without recompiling everything
    or explicitly providing the same options every time.
    
    This module also contains the auto-detection logic for figuring out
    the currently installed HDF5 version.
"""

from distutils.cmd import Command
import os
import os.path as op
import sys
import pickle

def loadpickle():
    """ Load settings dict from the pickle file """
    try:
        with open('h5config.pkl','rb') as f:
            cfg = pickle.load(f)
        if not isinstance(cfg, dict): raise TypeError
    except Exception:
        return {}
    return cfg


def savepickle(dct):
    """ Save settings dict to the pickle file """
    with open('h5config.pkl','wb') as f:
        pickle.dump(dct, f, protocol=0)


def validate_version(s):
    """ Ensure that s contains an X.Y.Z format version string, or ValueError.
    """
    try:
        tpl = tuple(int(x) for x in s.split('.'))
        if len(tpl) != 3: raise ValueError
    except Exception:
        raise ValueError("HDF5 version string must be in X.Y.Z format")


class EnvironmentOptions(object):

    """
        Convenience class representing the current environment variables.
    """
    
    def __init__(self):
        self.hdf5 = os.environ.get('HDF5_DIR')
        self.hdf5_version = os.environ.get('HDF5_VERSION')
        self.mpi = os.environ.get('HDF5_MPI') == "ON"
        if self.hdf5_version is not None:
            validate_version(self.hdf5_version)


class configure(Command):

    """
        Configure build options for h5py: custom path to HDF5, version of
        the HDF5 library, and whether MPI is enabled.
    
        Options come from the following sources, in order of priority:
        
        1. Current command-line options
        2. Old command-line options
        3. Current environment variables
        4. Old environment variables
        5. Autodetection
        
        When options change, the rebuild_required attribute is set, and
        may only be reset by calling reset_rebuild().  The custom build_ext
        command does this.s
    """

    description = "Configure h5py build options"

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
        """ Mark this configuration as built """
        dct = loadpickle()
        dct['rebuild'] = False
        savepickle(dct)
        
    def run(self):
        """ Distutils calls this when the command is run """
        
        env = EnvironmentOptions()
                
        # Step 1: determine if settings have changed and update cache
        
        oldsettings = {} if self.reset else loadpickle()
        dct = oldsettings.copy()
        
        # Only update settings which have actually been specified this
        # round; ignore the others (which have value None).
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
        if env.mpi is not None:
            dct['env_mpi'] = env.mpi

        self.rebuild_required = dct.get('rebuild') or dct != oldsettings
        
        # Corner case: rebuild if options reset, but only if they previously
        # had non-default values (to handle multiple resets in a row)
        if self.reset and any(loadpickle().values()):
            self.rebuild_required = True
            
        dct['rebuild'] = self.rebuild_required
        
        savepickle(dct)
    
        # Step 2: update public config attributes according to priority rules
          
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
            try:
                self.hdf5_version = autodetect_version(self.hdf5)
                print("Autodetected HDF5 %s" % self.hdf5_version)
            except Exception as e:
                sys.stderr.write("Autodetection skipped [%s]\n" % e)
                self.hdf5_version = '1.8.4'
                
        if self.mpi is None:
            self.mpi = oldsettings.get('cmd_mpi')
        if self.mpi is None:
            self.mpi = env.mpi
        if self.mpi is None:
            self.mpi = oldsettings.get('env_mpi')
                
        # Step 3: print the resulting configuration to stdout

        print('*' * 80)
        print(' ' * 23 + "Summary of the h5py configuration")
        print('')
        print("    Path to HDF5: " + repr(self.hdf5))
        print("    HDF5 Version: " + repr(self.hdf5_version))
        print("     MPI Enabled: " + repr(bool(self.mpi)))
        print("Rebuild Required: " + repr(bool(self.rebuild_required)))
        print('')
        print('*' * 80)


def autodetect_version(hdf5_dir=None):
    """
    Detect the current version of HDF5, and return X.Y.Z version string.

    Intended for Unix-ish platforms (Linux, OS X, BSD).
    Does not support Windows. Raises an exception if anything goes wrong.

    hdf5_dir: optional HDF5 install directory to look in (containing "lib")
    """

    import os
    import sys
    import os.path as op
    import re
    import ctypes
    from ctypes import byref

    import pkgconfig
    
    if sys.platform.startswith('darwin'):
        regexp = re.compile(r'^libhdf5.dylib')
    else:
        regexp = re.compile(r'^libhdf5.so')
        
    libdirs = ['/usr/local/lib', '/opt/local/lib']
    try:
        if pkgconfig.exists("hdf5"):
            libdirs.extend(pkgconfig.parse("hdf5")['library_dirs'])
    except EnvironmentError:
        pass
    if hdf5_dir is not None:
        libdirs.insert(0, op.join(hdf5_dir, 'lib'))

    path = None
    for d in libdirs:
        try:
            candidates = [x for x in os.listdir(d) if regexp.match(x)]
        except Exception:
            continue   # Skip invalid entries

        if len(candidates) != 0:
            candidates.sort(key=lambda x: len(x))   # Prefer libfoo.so to libfoo.so.X.Y.Z
            path = op.abspath(op.join(d, candidates[0]))
            break

    if path is None:
        path = "libhdf5.so"

    lib = ctypes.cdll.LoadLibrary(path)

    major = ctypes.c_uint()
    minor = ctypes.c_uint()
    release = ctypes.c_uint()

    lib.H5get_libversion(byref(major), byref(minor), byref(release))

    return "{0}.{1}.{2}".format(int(major.value), int(minor.value), int(release.value))
