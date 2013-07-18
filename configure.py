
"""
    Implements discovery of environment vars and command-line "superoptions".
    
    Also manages persistence via pickle files.

    Currently defined settings (Key: Value):

    "hdf5": <path to HDF5>
    "mpi": bool, should we build in mpi mode
    "hdf5_version": 3-tuple (major, minor, release)
"""

import os, sys


def printerr(what):

    sys.stderr.write(str(what)+'\n')
    sys.stderr.flush()


def loadpickle(name):
    """ Load object from pickle file, or None if it can't be opened """

    import pickle

    try:
        f = open(name,'r')
    except IOError:
        return None
    try:
        return pickle.load(f)
    except Exception:
        return None
    finally:
        f.close()


def savepickle(name, data):
    """ Save to pickle file, ignoring if it can't be written """
    import pickle
    try:
        f = open(name, 'wb')
    except IOError:
        return
    try:
        pickle.dump(data, f)
    finally:
        f.close()


def parse_hdf5_version(vers):
    """ Split HDF5 version string X.Y.Z into a tuple.  ValueError on failure.
    """

    try:
        vers = tuple(int(x) for x in vers.split('.'))
        if len(vers) != 3:
            raise ValueError
    except Exception:
        raise ValueError("Illegal value for HDF5 version")

    return vers


def scrape_eargs():
    """ Locate settings in environment vars """

    settings = {}

    hdf5 = os.environ.get("HDF5_DIR", '')
    if hdf5 != '':
        settings['hdf5'] = hdf5

    api = os.environ.get("HDF5_API", '')
    if api != '':
        printerr("HDF5_API environment variable ignored (Support for HDF5 1.6 dropped)")

    vers = os.environ.get("HDF5_VERSION", '')
    if vers != '':
        try:
            vers = parse_hdf5_version(vers)
        except ValueError:
            printerr("Invalid format for $HDF5_VERSION (must be X.Y.Z)")
        else:
            settings['hdf5_version'] = vers

    return settings

def scrape_cargs():
    """ Locate settings in command line or pickle file """

    settings = loadpickle('h5py_config.pickle')
    if settings is None: settings = {}

    for arg in sys.argv[:]:

        if arg.find('--hdf5=') == 0:
            hdf5 = arg.split('=')[-1]
            if hdf5.lower() == 'default':
                settings.pop('hdf5', None)
            else:
                settings['hdf5'] = hdf5
            sys.argv.remove(arg)

        if arg.find('--mpi') == 0:
            if arg in ('--mpi','--mpi=yes'):
                settings['mpi'] = True
            elif arg == '--mpi=no':
                settings['mpi'] = False
            else:
                raise ValueError("Invalid option for --mpi (--mpi or --mpi=[yes|no])")
            sys.argv.remove(arg)

        if arg.find('--hdf5-version=') == 0:
            vers = arg.split('=')[-1]
            if vers.lower() == 'default':
                settings.pop('hdf5_version', None)
            else:
                try:
                    vers = parse_hdf5_version(vers)
                    settings['hdf5_version'] = vers
                except ValueError:
                    raise ValueError('Illegal option "%s" for --hd5-version (must be "default" or "X.Y.Z")' % vers)
            sys.argv.remove(arg)

        if arg.find('--api=') == 0:
            printerr("--api option ignored (Support for HDF5 1.6 dropped)")
            sys.argv.remove(arg)

    savepickle('h5py_config.pickle', settings)
    return settings


# --- Autodetection of HDF5 library via ctypes --------------------------------

def autodetect(libdirs):
    """
    Detect the current version of HDF5, and return it as a tuple
    (major, minor, release).

    If the version can't be determined, prints error information to stderr
    and returns None.

    libdirs: list of library paths to search for libhdf5.so.  The first one
    which can be successfully loaded will be used.
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

        printerr("Autodetected HDF5 version %s" % ".".join(str(x) for x in vers))

        return vers

    except Exception as e:

        printerr(e)
        return None

