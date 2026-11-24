"""Exercise the SWMR example against actual Linux inotify events."""
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='requires Linux inotify')


def test_swmr_inotify_reader_observes_updates_and_writer_close(tmp_path):
    pytest.importorskip('watchdog')
    example = Path(__file__).resolve().parents[1] / 'swmr_inotify_example.py'
    filename = tmp_path / 'swmr.h5'
    messages = queue.Queue()
    output = []

    def wait_for(message):
        deadline = time.monotonic() + 20
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(f'Timed out waiting for {message!r}. Output: {output!r}')
            try:
                line = messages.get(timeout=remaining)
            except queue.Empty:
                pytest.fail(f'Timed out waiting for {message!r}. Output: {output!r}')
            output.append(line)
            if message in line:
                return

    with h5py.File(filename, 'w', libver='latest') as writer:
        dataset = writer.create_dataset('data', shape=(0,), maxshape=(None,), dtype='i')
        writer.swmr_mode = True
        reader = subprocess.Popen(
            [sys.executable, str(example), str(filename), 'data'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        def collect_output():
            for line in reader.stdout:
                messages.put(line)

        collector = threading.Thread(target=collect_output, daemon=True)
        collector.start()
        try:
            wait_for('Watching ')
            for size in (3, 5):
                previous_size = dataset.shape[0]
                dataset.resize((size,))
                dataset[previous_size:] = range(previous_size, size)
                dataset.flush()
                wait_for(f'Read data shape: ({size},)')
                wait_for(f'Reading out dataset [{previous_size}:{size}]')
        except BaseException:
            reader.terminate()
            reader.wait(timeout=10)
            raise

    try:
        wait_for('File writer closed file')
        assert reader.wait(timeout=20) == 0
    finally:
        if reader.poll() is None:
            reader.terminate()
            reader.wait(timeout=10)
        reader.stdout.close()
        collector.join(timeout=10)
