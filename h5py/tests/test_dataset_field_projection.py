# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License: Standard 3-clause BSD; see "license.txt" for full license terms
# and contributor agreement.

from contextlib import closing

import h5py
import numpy as np
import pytest

from h5py.tests.common import make_name


@pytest.mark.parametrize('kind', ['numeric', 'string'])
@pytest.mark.parametrize('access', ['fields', 'mixed', 'scalar', 'fancy', 'direct'])
def test_projection_does_not_read_unused_vlen(tmp_path, kind, access):
    """An unselected field must not be read into a temporary vlen buffer."""
    vlen_dtype = (h5py.vlen_dtype('u1') if kind == 'numeric'
                  else h5py.string_dtype())
    dtype = np.dtype([('time', '<u8'), ('unused', vlen_dtype)])
    data = np.zeros(2, dtype=dtype)
    data['time'] = [11, 22]
    payload = np.arange(4, dtype='u1') if kind == 'numeric' else b'abcd'
    data['unused'] = [payload, payload]
    path = tmp_path / make_name('projection.h5')
    with h5py.File(path, 'w') as file:
        dataset = file.create_dataset('events', data=data, chunks=(2,))
        file.flush()
        with closing(file.id.get_create_plist()) as fcpl:
            address_size, _ = fcpl.get_sizes()
        with closing(dataset.id.get_type()) as disk_type:
            offset = disk_type.get_member_offset(1)
        filter_mask, chunk = dataset.id.read_direct_chunk((0,))
        assert filter_mask == 0
        # H5Tget_size reports the in-memory size of vlen strings, which is
        # different from their on-disk heap descriptor size.
        row_size = len(chunk) // 2
        assert len(chunk) == 2 * (offset + 8 + address_size)
        chunk = bytearray(chunk)
        for row in range(2):
            # Native HDF5 stores vlen data as a 4-byte sequence length, a
            # global heap address, and a 4-byte object index. Use an invalid
            # address in the unused field to detect any attempt to read it,
            # without depending on allocator behaviour or an RSS threshold.
            start = row * row_size + offset + 4
            chunk[start:start + address_size] = (1).to_bytes(address_size, 'little')
        dataset.id.write_direct_chunk((0,), chunk)

    with h5py.File(path, 'r') as file:
        dataset = file['events']
        # Confirm the fixture detects a read of the corrupted vlen data.
        with pytest.raises(OSError):
            dataset[:]
        if access == 'fields':
            output = dataset.fields('time')[:]
        elif access == 'mixed':
            output = dataset[:, 'time']
        elif access == 'scalar':
            output = dataset[0, 'time']
        elif access == 'fancy':
            output = dataset.fields('time')[[0, 1]]
        else:
            output = np.zeros(2, dtype=[('time', '<u8')])
            dataset.read_direct(output)
            output = output['time']
        np.testing.assert_array_equal(output, 11 if access == 'scalar' else [11, 22])


@pytest.mark.parametrize('kind', ['numeric', 'string'])
@pytest.mark.parametrize('fields', [('time',), ('samples', 'selected'),
                                  ('selected', 'time')])
def test_projection_preserves_selected_fields(tmp_path, kind, fields):
    vlen_dtype = (h5py.vlen_dtype('i4') if kind == 'numeric'
                  else h5py.string_dtype())
    dtype = np.dtype({'names': ['unused', 'time', 'selected', 'samples'],
                      'formats': [vlen_dtype, '>u8', vlen_dtype, ('<f4', (2,))],
                      'offsets': [0, 16, 32, 48], 'itemsize': 64})
    data = np.zeros(5, dtype=dtype)
    data['time'] = np.arange(5)
    data['samples'] = np.arange(10, dtype='f4').reshape(5, 2)
    for row in range(5):
        payload = (np.arange(row + 1, dtype='i4') if kind == 'numeric'
                   else f'value{row}'.encode())
        data['unused'][row] = payload
        data['selected'][row] = payload
    with h5py.File(tmp_path / make_name('selected.h5'), 'w') as file:
        dataset = file.create_dataset('events', data=data)
        for selection in (0, slice(None, None, 2), [0, 2, 4]):
            output = dataset.fields(list(fields))[selection]
            assert output.dtype.names == fields
            expected = data[selection]
            for field in fields:
                if field == 'selected' and selection != 0:
                    for actual, value in zip(output[field], expected[field]):
                        np.testing.assert_array_equal(actual, value)
                else:
                    np.testing.assert_array_equal(output[field], expected[field])
        np.testing.assert_array_equal(dataset.fields('time')[0:0], [])
        with pytest.raises(ValueError):
            dataset.fields('missing')
        # The error class for an invalid selection varies with HDF5 version.
        with pytest.raises((ValueError, OSError), match='not within extent'):
            dataset.fields('time')[[5]]


@pytest.mark.parametrize('kind', ['numeric', 'string'])
@pytest.mark.parametrize('empty', [False, True])
def test_projection_without_matching_fields(tmp_path, kind, empty):
    vlen_dtype = (h5py.vlen_dtype('u1') if kind == 'numeric'
                  else h5py.string_dtype())
    data = np.zeros(2, dtype=[('number', 'i4'), ('unused', vlen_dtype)])
    data['number'] = [12, 34]
    data['unused'] = ([np.arange(3, dtype='u1')] * 2 if kind == 'numeric'
                      else [b'abc', b'def'])
    with h5py.File(tmp_path / make_name('unmatched.h5'), 'w') as file:
        dataset = file.create_dataset('events', data=data)
        output = np.full(2, 7, dtype=[('missing', 'i4')])
        with closing(h5py.h5t.py_create(output.dtype)) as memory_type, \
             closing(dataset.id.get_space()) as file_space, \
             closing(h5py.h5s.create_simple((2,))) as memory_space:
            if empty:
                memory_space.select_none()
                file_space.select_none()
            dataset.id.read(memory_space, file_space, output, mtype=memory_type)
        np.testing.assert_array_equal(output['missing'], [7, 7])


@pytest.mark.parametrize('kind', ['numeric', 'string'])
def test_projection_preserves_conversion_error(tmp_path, kind):
    vlen_dtype = (h5py.vlen_dtype('u1') if kind == 'numeric'
                  else h5py.string_dtype())
    data = np.zeros(2, dtype=[('number', 'i4'), ('unused', vlen_dtype)])
    data['number'] = [12, 34]
    data['unused'] = ([np.arange(3, dtype='u1')] * 2 if kind == 'numeric'
                      else [b'abc', b'def'])
    with h5py.File(tmp_path / make_name('conversion.h5'), 'w') as file:
        dataset = file.create_dataset('events', data=data)
        output = np.zeros(2, dtype=[('number', 'S4')])
        with closing(h5py.h5t.py_create(output.dtype)) as memory_type, \
             closing(dataset.id.get_space()) as file_space, \
             closing(h5py.h5s.create_simple((2,))) as memory_space:
            with pytest.raises(TypeError, match='no appropriate function'):
                dataset.id.read(memory_space, file_space, output, mtype=memory_type)
