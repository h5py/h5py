from paver.easy import *

@task
def release_unix():
    sh('python setup.py clean')
    sh('rm -f h5py_config.pickle')
    sh('rm -f dist/*.tar.gz')
    sh('python setup.py build --hdf5-version=1.8.4 --mpi=no')
    sh('python setup.py test')
    sh('python setup.py sdist')
    print("Unix release done.  Distribution tar file is in dist/")

    