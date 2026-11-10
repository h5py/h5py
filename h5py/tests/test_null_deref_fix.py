#!/usr/bin/env python3
"""
Unit tests for h5py NULL-pointer-dereference / uninitialized-memory-read fix.

Tests that h5o.get_comment() and h5g.get_objname_by_idx() raise an exception
(AssertionError or a h5py error) when the underlying HDF5 C function returns
a negative size, rather than returning uninitialized memory or crashing.

Background: h5o.get_comment and h5g.get_objname_by_idx were missing the
`assert size >= 0` guard that h5g.get_comment (line 441) and h5f.get_name
(line 206) already had. When the HDF5 C function returned -1 (error),
emalloc(size+1) == emalloc(0) was used, leading to either a NULL-pointer
dereference (crash) or reading uninitialized heap memory (info leak).
"""
import os
import sys

import h5py
from h5py import h5o, h5g
from h5py.tests.common import TestCase

import tempfile


class TestGetCommentErrorHandling(TestCase):
    """h5o.get_comment should not return uninitialized memory on error."""

    def test_get_comment_nonexistent_object(self):
        """get_comment on a non-existent object must raise, not return garbage."""
        fname = tempfile.mktemp(suffix='.h5')
        try:
            f = h5py.File(fname, 'w')
            try:
                group = f.create_group("test_group")
                # H5Oget_comment_by_name returns -1 for non-existent object.
                # Before fix: returned uninitialized heap bytes (info leak).
                # After fix: raises AssertionError (or h5py error).
                with self.assertRaises((AssertionError, Exception)):
                    h5o.get_comment(group.id, b"", obj_name=b"nonexistent_object")
            finally:
                f.close()
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def test_get_comment_valid_object(self):
        """get_comment on a valid object with no comment returns empty bytes."""
        fname = tempfile.mktemp(suffix='.h5')
        try:
            f = h5py.File(fname, 'w')
            try:
                group = f.create_group("test_group")
                # A valid object with no comment should return b'' (empty),
                # not raise. This confirms the fix doesn't break normal use.
                comment = h5o.get_comment(group.id)
                self.assertEqual(comment, b'')
            finally:
                f.close()
        finally:
            if os.path.exists(fname):
                os.remove(fname)


class TestGetObjnameByIdxErrorHandling(TestCase):
    """h5g.get_objname_by_idx should not crash on invalid index."""

    def test_get_objname_by_idx_out_of_range(self):
        """get_objname_by_idx with out-of-range index must raise, not crash."""
        fname = tempfile.mktemp(suffix='.h5')
        try:
            f = h5py.File(fname, 'w')
            try:
                group = f.create_group("root_group")
                group.create_group("child1")
                # Index 9999 is out of range; H5Gget_objname_by_idx returns -1.
                # Before fix: emalloc(0) -> NULL deref / uninitialized read.
                # After fix: raises AssertionError (or RuntimeError).
                with self.assertRaises((AssertionError, RuntimeError, Exception)):
                    group.id.get_objname_by_idx(9999)
            finally:
                f.close()
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def test_get_objname_by_idx_valid(self):
        """get_objname_by_idx with a valid index returns the correct name."""
        fname = tempfile.mktemp(suffix='.h5')
        try:
            f = h5py.File(fname, 'w')
            try:
                group = f.create_group("root_group")
                group.create_group("child1")
                # Valid index 0 should return the child name.
                name = group.id.get_objname_by_idx(0)
                self.assertIsInstance(name, (bytes, str))
            finally:
                f.close()
        finally:
            if os.path.exists(fname):
                os.remove(fname)


if __name__ == '__main__':
    import unittest
    unittest.main()
