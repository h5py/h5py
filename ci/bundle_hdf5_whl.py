"""Bundle HDF5 DLLs into an h5py wheel on Windows

This is meant to do something like auditwheel on Linux & delocate on Mac,
but h5py-specific.
"""
from base64 import urlsafe_b64encode
from contextlib import contextmanager
from glob import glob
import hashlib
import os
import os.path as osp
import shutil
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

def find_dlls():
    hdf5_path = os.environ.get("HDF5_DIR")
    print("HDF5_DIR", hdf5_path)
    yield from glob(os.path.join(hdf5_path, 'lib', 'hdf*.dll'))
    zlib_root = os.environ.get("ZLIB_ROOT")
    if zlib_root:
        print("ZLIB_ROOT", zlib_root)
        yield os.path.join(zlib_root, 'bin_release', 'zlib.dll')

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as src:
        while True:
            buf = src.read(1024 * 8)
            if not buf:
                break
            h.update(buf)

    return urlsafe_b64encode(h.digest()).decode('ascii').rstrip('=')

def find_wheels():
    wheelhouse_dir = sys.argv[1]
    return glob(osp.join(wheelhouse_dir, '*.whl'))

@contextmanager
def modify_zip(zip_file):
    with tempfile.TemporaryDirectory() as td:
        with ZipFile(zip_file, 'r') as zf:
            zf.extractall(path=td)
        yield td

        with ZipFile(zip_file, 'w', compression=ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(td):
                for f in sorted(files):
                    path = osp.join(root, f)
                    zf.write(path, arcname=osp.relpath(path, td))

                dirs.sort()

def bundle(whl_file):
    print("Adding DLLs to", whl_file)

    with modify_zip(whl_file) as td:
        # Find & read RECORD file
        records = glob(osp.join(td, '*.dist-info', 'RECORD'))
        assert len(records) == 1, records
        record_f = records[0]
        with open(record_f, encoding='utf-8') as f:
            record = f.read().strip() + '\n'

        # Copy DLLs & add them to RECORD
        for dll in find_dlls():
            size = os.stat(dll).st_size
            sha = file_sha256(dll)
            dest = 'h5py/' + os.path.basename(dll)
            print(f"{dest} ({size} bytes)")
            shutil.copy2(dll, osp.join(td, dest))

            record += f'{dest},sha256={sha},{size}\n'

        print("Writing modified", record_f)
        with open(record_f, 'w', encoding='utf-8') as f:
            f.write(record)


def main():
    if not sys.platform.startswith('win'):
        print("Non-windows platform, skipping bundle_hdf5_whl.py")
        return

    for whl_file in find_wheels():
        bundle(whl_file)


if __name__ == '__main__':
    main()
