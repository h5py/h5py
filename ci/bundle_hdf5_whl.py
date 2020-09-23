"""Bundle HDF5 DLLs into an h5py wheel on Windows

This is meant to do something like auditwheel on Linux & delocate on Mac,
but h5py-specific.
"""
from base64 import urlsafe_b64encode
from glob import glob
import hashlib
import os
import sys
from zipfile import ZipFile, ZIP_DEFLATED

def find_dlls():
    hdf5_path = os.environ.get("HDF5_DIR")
    print("HDF5_DIR", hdf5_path)
    return glob(os.path.join(hdf5_path, 'lib', '*.dll'))

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
    return glob(os.path.join(wheelhouse_dir, '*.whl'))

def bundle(whl_file):
    print("Adding DLLs to", whl_file)

    with ZipFile(whl_file, 'a', compression=ZIP_DEFLATED) as zf:
        record_zinfos = {zi for zi in zf.infolist() if zi.filename.endswith('RECORD')}
        assert len(record_zinfos) == 1, record_zinfos
        record_zinfo = record_zinfos.pop()
        record = zf.read(record_zinfo).strip() + b'\n'

        for dll in find_dlls():
            size = os.stat(dll).st_size
            sha = file_sha256(dll)
            dest = 'h5py/' + os.path.basename(dll)
            print(f"{dest} ({size} bytes)")
            zf.write(dll, dest)

            record += f'{dest},sha256={sha},{size}\n'.encode('utf-8')

        print("Writing modified", record_zinfo.filename)
        zf.writestr(record_zinfo, record)


def main():
    if not sys.platform.startswith('win'):
        print("Non-windows platform, skipping bundle_hdf5_whl.py")
        return

    for whl_file in find_wheels():
        bundle(whl_file)


if __name__ == '__main__':
    main()
