Exposing HDF5 functions
-----------------------

* Exposes remaining information from `H5O_info_t` struct such as access, modification, change, and
birth time. Also exposes field providing number of attributes attached to an object. Expands object
header metadata struct `H5O_hdr_info_t`, `hdr` field of `H5O_info_t`, to provide number of chunks and
flags set for object header. Lastly, adds `meta_size` field from `H5O_info_t` struct that provides
two fields, `attr` which is the storage overhead of any attached attributes, and `obj` which is
storage overhead required for chunk storage. The last two fields added can be useful for determining
the storage overhead incurred from various data layout/chunked strategies, and for obtaining information
such as that provided by `h5stat`.
