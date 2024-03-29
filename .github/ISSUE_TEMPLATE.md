To assist reproducing bugs, please include the following:
 * Operating System (e.g. Windows 10, MacOS 10.11, Ubuntu 20.04.2 LTS, CentOS 7)
 * Python version (e.g. 3.10)
 * Where Python was acquired (e.g. system Python on MacOS or Linux, Anaconda on
   Windows)
 * h5py version (e.g. 3.9)
 * HDF5 version (e.g. 1.12.2)
 * The full traceback/stack trace shown (if it appears)

`h5py.version.info` contains the needed versions, which can be displayed by
```
python -c 'import h5py; print(h5py.version.info)'
```
where `python` should be substituted for the path to python used to install
`h5py` with.
