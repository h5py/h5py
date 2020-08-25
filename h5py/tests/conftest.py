import os

import h5py
import pytest


@pytest.fixture()
def writable_file(tmp_path):
    with h5py.File(tmp_path / 'test.h5', 'w') as f:
        yield f


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "subprocess(env=None): mark test to run in subprocess"
    )


def pytest_runtest_setup(item):
    if "subprocess" in item.iter_markers():
        if os.environ.get("CYTHON_COVERAGE"):
            pytest.skip(
                "subprocess tests cannot be run with CYTHON_COVERAGE enabled."
            )
