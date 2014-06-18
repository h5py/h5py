from paver.easy import *

@task
def release_unix():
    sh('python setup.py clean')
    sh('rm -f h5py_config.pickle')
    sh('python setup.py build --hdf5-version=1.8.4 --mpi=no')
    sh('python setup.py test')
    sh('python setup.py sdist')
    print("Unix release done.  Distribution tar file is in dist/")

@task
def release_windows():
    for pyver in (26, 27, 32, 33):
        exe = r'C:\Python%d\Python.exe' % pyver
        hdf5 = r'c:\hdf5\Python%d' % pyver
        sh('%s setup.py clean' % exe)
        sh('%s api_gen.py' % exe)
        sh('%s setup.py build -f --hdf5-version=1.8.4 --hdf5=%s' % (exe, hdf5))
        sh('%s setup.py test --hdf5-version=1.8.4 --hdf5=%s' % (exe, hdf5))
        sh('%s setup.py bdist_wininst --hdf5-version=1.8.4 --hdf5=%s' % (exe, hdf5))
    print ("Windows exe release done.  Distribution files are in dist/")
    
@task
@consume_args
def git_summary(options):
    sh('git log --no-merges --pretty=oneline --abbrev-commit %s..HEAD'%options.args[0])
    sh('git shortlog -s -n %s..HEAD'%options.args[0])
