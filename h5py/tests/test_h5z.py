from ctypes import (
    addressof,
    c_char_p,
    c_int,
    c_long,
    c_uint,
    c_void_p,
    CFUNCTYPE,
    POINTER,
    Structure,
)

from h5py import h5z


# Type of dilter callback function of H5Z_class2_t
H5ZFuncT = CFUNCTYPE(
    c_long,  # restype
    # argtypes
    c_uint,  # flags
    c_long,  # cd_nelemts
    POINTER(c_uint),  # cd_values
    c_long,  # nbytes
    POINTER(c_long),  # buf_size
    POINTER(c_void_p),  # buf
)


class H5ZClass2T(Structure):
    """H5Z_class2_t structure defining a filter"""

    _fields_ = [
        ("version", c_int),
        ("id_", c_int),
        ("encoder_present", c_uint),
        ("decoder_present", c_uint),
        ("name", c_char_p),
        ("can_apply", c_void_p),
        ("set_local", c_void_p),
        ("filter_", H5ZFuncT),
    ]


def test_register_filter():
    filter_id = 256  # Test ID

    @H5ZFuncT
    def failing_filter_callback(flags, cd_nelemts, cd_values, nbytes, buf_size, buf):
        return 0

    dummy_filter_class = H5ZClass2T(
        version=h5z.CLASS_T_VERS,
        id_=filter_id,
        encoder_present=1,
        decoder_present=1,
        name=b"dummy filter",
        can_apply=None,
        set_local=None,
        filter_=failing_filter_callback,
    )

    h5z.register_filter(addressof(dummy_filter_class))

    try:
        assert h5z.filter_avail(filter_id)
        filter_flags = h5z.get_filter_info(filter_id)
        assert (
            filter_flags
            == h5z.FILTER_CONFIG_ENCODE_ENABLED | h5z.FILTER_CONFIG_DECODE_ENABLED
        )
    finally:
        h5z.unregister_filter(filter_id)

    assert not h5z.filter_avail(filter_id)
