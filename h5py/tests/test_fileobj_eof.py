"""Exceptions raised while a Python file object reports its end position."""
import io

import h5py
import pytest


@pytest.mark.parametrize('method', ['seek', 'tell'])
@pytest.mark.parametrize('exception_type', [ValueError, NotImplementedError])
@pytest.mark.parametrize('libver', ['earliest', 'latest'])
def test_fileobj_eof_exception(method, exception_type, libver):
    error = exception_type('file-object EOF error')

    class BrokenBytesIO(io.BytesIO):
        fail = False
        failed_calls = 0

        def seek(self, offset, whence=0):
            if self.fail and method == 'seek':
                self.failed_calls += 1
                raise error
            return super().seek(offset, whence)

        def tell(self):
            if self.fail and method == 'tell':
                self.failed_calls += 1
                raise error
            return super().tell()

    fileobj = BrokenBytesIO()
    with h5py.File(fileobj, 'w', libver=libver) as f:
        f['data'] = list(range(10))
    try:
        fileobj.fail = True
        try:
            with pytest.raises(exception_type) as exc:
                h5py.File(fileobj, 'r', libver=libver)
        finally:
            fileobj.fail = False
        assert exc.value is error
        assert fileobj.failed_calls == 1

        with h5py.File(fileobj, 'r') as f:
            assert list(f['data'][:]) == list(range(10))
    finally:
        fileobj.close()
        # Break the cycle through the callback and failed-open traceback.
        error.__traceback__ = None


@pytest.mark.parametrize('libver', ['earliest', 'latest'])
def test_fileobj_open_with_handled_exception(libver):
    with io.BytesIO() as fileobj:
        with h5py.File(fileobj, 'w', libver=libver) as f:
            f['data'] = [1, 2, 3]
        try:
            raise ValueError('handled outer exception')
        except ValueError:
            with h5py.File(fileobj, 'r') as f:
                assert list(f['data'][:]) == [1, 2, 3]
