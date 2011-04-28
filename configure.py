
"""
    Implements discovery of environment vars and command-line "superoptions".
    
    Also manages persistence via pickle files.

    Currently defined settings (Key: Value):

    "hdf5": <path to HDF5>
"""

import os, sys
import warnings

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

def scrape_eargs():
    """ Locate settings in environment vars """
    settings = {}

    hdf5 = os.environ.get("HDF5_DIR", '')
    if hdf5 != '':
        settings['hdf5'] = hdf5

    api = os.environ.get("HDF5_API", '')
    if api != '':
        warnings.warn("HDF5_API environment variable ignored (Support for HDF5 1.6 dropped)")

    return settings

def scrape_cargs():
    """ Locate settings in command line or pickle file """
    settings = loadpickle('h5py_config.pickle')
    if settings is None:  settings = {}
    for arg in sys.argv[:]:
        if arg.find('--hdf5=') == 0:
            hdf5 = arg.split('=')[-1]
            if hdf5.lower() == 'default':
                settings.pop('hdf5', None)
            else:
                settings['hdf5'] = hdf5
            sys.argv.remove(arg)
        if arg.find('--api=') == 0:
            warnings.warn("--api option ignored (Support for HDF5 1.6 dropped)")
            sys.argv.remove(arg)
    savepickle('h5py_config.pickle', settings)
    return settings


    
