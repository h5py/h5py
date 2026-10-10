#!/usr/bin/env python3
"""Verify that the documented exceptions in ``File.__init__`` are the ones
the function actually raises, and that the docstring says so.

Two kinds of test live here, and the distinction is the point:

* **Behaviour** tests assert what h5py does. They are the ground truth the
  documentation is checked against, and they are listed first.
* A **documentation** test asserts that the docstring mentions each of those
  exceptions. Without this one, the behaviour tests pass whether or not the
  docstring was ever edited, so they cannot be evidence about the change.
"""
import inspect
import pytest
import h5py
import tempfile
import os


def _tmpfile():
    return os.path.join(tempfile.mkdtemp(), "test.h5")


class TestFileInitBehaviour:
    """Ground truth: what ``File.__init__`` raises, measured on real h5py."""

    def test_mode_invalid_raises_valueerror(self):
        with pytest.raises(ValueError):
            h5py.File(_tmpfile(), "z")

    def test_missing_file_raises_filenotfounderror(self):
        with pytest.raises(FileNotFoundError):
            h5py.File("/nonexistent/dir/f.h5", "r")

    def test_missing_file_readwrite_raises_filenotfounderror(self):
        with pytest.raises(FileNotFoundError):
            h5py.File("/nonexistent/dir/f.h5", "r+")

    def test_x_mode_existing_raises_fileexistserror(self):
        p = _tmpfile()
        with h5py.File(p, "w"):
            pass
        with pytest.raises(FileExistsError):
            h5py.File(p, "x")

    def test_w_dash_mode_existing_raises_fileexistserror(self):
        # 'w-' maps to the same exclusive-create flag as 'x'.
        p = _tmpfile()
        with h5py.File(p, "w"):
            pass
        with pytest.raises(FileExistsError):
            h5py.File(p, "w-")


class TestFileInitDocstring:
    """The docstring must state the exceptions measured above.

    This is the test that fails when the documentation is wrong or absent.
    The behaviour tests pass either way; this one cannot.
    """

    def _doc(self):
        return inspect.getdoc(h5py.File.__init__) or ""

    def test_docstring_documents_valueerror(self):
        assert "ValueError" in self._doc()

    def test_docstring_documents_filenotfounderror(self):
        assert "FileNotFoundError" in self._doc()

    def test_docstring_documents_fileexistserror(self):
        assert "FileExistsError" in self._doc()

    def test_docstring_does_not_claim_a_typeerror(self):
        # File.__init__ raises no TypeError itself; unknown keywords are
        # rejected deeper in the stack. Claiming one here misattributes it.
        assert "TypeError" not in self._doc()
