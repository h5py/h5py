"""
Compatibility module for high-level h5py
"""
import sys
import six


try:
    from os import fspath
except ImportError:
    def fspath(path):
        """
        Return the string representation of the path.
        If str or bytes is passed in, it is returned unchanged.
        This code comes from PEP 519, modified to support earlier versions of
        python.

        This is required for python < 3.6.
        """
        if isinstance(path, (six.text_type, six.binary_type)):
            return path

        # Work from the object's type to match method resolution of other magic
        # methods.
        path_type = type(path)
        try:
            return path_type.__fspath__(path)
        except AttributeError:
            if hasattr(path_type, '__fspath__'):
                raise
            try:
                import pathlib
            except ImportError:
                pass
            else:
                if isinstance(path, pathlib.PurePath):
                    return six.text_type(path)

            raise TypeError("expected str, bytes or os.PathLike object, not "
                            + path_type.__name__)

            
try:
    from os import fsencode as os_fsencode
    from os import fsdecode as os_fsdecode
except ImportError:
    # This is from python 3.5 stdlib (hence lacks PEP 519 changes)
    # This was introduced into python 3.2, so python < 3.2 does not have this
    # Effectively, this is only required for python 2.6 and 2.7, and can be removed
    # once support for them is dropped
    def _fscodec():
        encoding = sys.getfilesystemencoding()
        if encoding == 'mbcs':
            errors = 'strict'
        else:
            try:
                from codecs import lookup_error
                lookup_error('surrogateescape')
            except LookupError:
                errors = 'strict'
            else:
                errors = 'surrogateescape'

        def fsencode(filename):
            """
            Encode filename to the filesystem encoding with 'surrogateescape' error
            handler, return bytes unchanged. On Windows, use 'strict' error handler if
            the file system encoding is 'mbcs' (which is the default encoding).
            """
            if isinstance(filename, six.binary_type):
                return filename
            elif isinstance(filename, six.text_type):
                return filename.encode(encoding, errors)
            else:
                raise TypeError("expect bytes or str, not %s" % type(filename).__name__)

        def fsdecode(filename):
            """
            Decode filename from the filesystem encoding with 'surrogateescape' error
            handler, return str unchanged. On Windows, use 'strict' error handler if
            the file system encoding is 'mbcs' (which is the default encoding).
            """
            if isinstance(filename, six.text_type):
                return filename
            elif isinstance(filename, six.binary_type):
                return filename.decode(encoding, errors)
            else:
                raise TypeError("expect bytes or str, not %s" % type(filename).__name__)

        return fsencode, fsdecode

    os_fsencode, os_fsdecode = _fscodec()
    del _fscodec

# Python 3.6 change Windows encoding to UTF-8 (See PEP 529) but HDF library still use 'mcbs'
if sys.platform == 'win32' and sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    # This is from python 3.6.0 stdlib, with encoding and errors changed to 'mbcs' and 'strict'
    def fsencode(filename):
        """Encode filename (an os.PathLike, bytes, or str) to the filesystem
        encoding with 'surrogateescape' error handler, return bytes unchanged.
        On Windows, use 'strict' error handler if the file system encoding is
        'mbcs' (which is the default encoding).
        """
        filename = fspath(filename)  # Does type-checking of `filename`.
        if isinstance(filename, str):
            return filename.encode('mbcs', 'strict')
        else:
            return filename

    def fsdecode(filename):
        """Decode filename (an os.PathLike, bytes, or str) from the filesystem
        encoding with 'surrogateescape' error handler, return str unchanged. On
        Windows, use 'strict' error handler if the file system encoding is
        'mbcs' (which is the default encoding).
        """
        filename = fspath(filename)  # Does type-checking of `filename`.
        if isinstance(filename, bytes):
            return filename.decode('mbcs', 'strict')
        else:
            return filename

# In all other cases, use stdlib os.fsencode/os.fsdecode
else:
    fsencode = os_fsencode
    fsdecode = os_fsdecode
