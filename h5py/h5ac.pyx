# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.


"""
    Low-level HDF5 "H5AC" cache configuration interface.
"""


cdef class CacheConfig:
    """Represents H5AC_cache_config_t objects

    """

    #cdef H5AC_cache_config_t cache_config
    #     /* general configuration fields: */
    def __cinit__(self):
        self.cache_config.version = H5AC__CURR_CACHE_CONFIG_VERSION

    property version:
        def __get__(self):
            return self.cache_config.version
        def __set__(self, int val):
            self.cache_config.version = val

    property rpt_fcn_enabled:
        def __get__(self):
            return self.cache_config.rpt_fcn_enabled
        def __set__(self, hbool_t val):
            self.cache_config.rpt_fcn_enabled = val

    property evictions_enabled:
        def __get__(self):
            return self.cache_config.evictions_enabled
        def __set__(self, hbool_t val):
            self.cache_config.evictions_enabled = val

    property set_initial_size:
        def __get__(self):
            return self.cache_config.set_initial_size
        def __set__(self, hbool_t val):
            self.cache_config.set_initial_size = val

    property initial_size:
        def __get__(self):
            return self.cache_config.initial_size
        def __set__(self, size_t val):
            self.cache_config.initial_size = val

    property min_clean_fraction:
        def __get__(self):
            return self.cache_config.min_clean_fraction
        def __set__(self, double val):
            self.cache_config.min_clean_fraction = val

    property max_size:
        def __get__(self):
            return self.cache_config.max_size
        def __set__(self, size_t val):
            self.cache_config.max_size = val

    property min_size:
        def __get__(self):
            return self.cache_config.min_size
        def __set__(self, size_t val):
            self.cache_config.min_size = val

    property epoch_length:
        def __get__(self):
            return self.cache_config.epoch_length
        def __set__(self, long int val):
            self.cache_config.epoch_length = val

    #    /* size increase control fields: */
    property incr_mode:
        def __get__(self):
            return self.cache_config.incr_mode
        def __set__(self, H5C_cache_incr_mode val):
            self.cache_config.incr_mode = val

    property lower_hr_threshold:
        def __get__(self):
            return self.cache_config.lower_hr_threshold
        def __set__(self, double val):
            self.cache_config.lower_hr_threshold = val

    property increment:
        def __get__(self):
            return self.cache_config.increment
        def __set__(self, double val):
            self.cache_config.increment = val

    property apply_max_increment:
        def __get__(self):
            return self.cache_config.apply_max_increment
        def __set__(self, hbool_t val):
            self.cache_config.apply_max_increment = val

    property max_increment:
        def __get__(self):
            return self.cache_config.max_increment
        def __set__(self, size_t val):
            self.cache_config.max_increment = val

    property flash_incr_mode:
        def __get__(self):
            return self.cache_config.flash_incr_mode
        def __set__(self, H5C_cache_flash_incr_mode val):
            self.cache_config.flash_incr_mode = val

    property flash_multiple:
        def __get__(self):
            return self.cache_config.flash_multiple
        def __set__(self, double val):
            self.cache_config.flash_multiple = val

    property flash_threshold:
        def __get__(self):
            return self.cache_config.flash_threshold
        def __set__(self, double val):
            self.cache_config.flash_threshold = val

    # /* size decrease control fields: */
    property decr_mode:
        def __get__(self):
            return self.cache_config.decr_mode
        def __set__(self, H5C_cache_decr_mode val):
            self.cache_config.decr_mode = val

    property upper_hr_threshold:
        def __get__(self):
            return self.cache_config.upper_hr_threshold
        def __set__(self, double val):
            self.cache_config.upper_hr_threshold = val

    property decrement:
        def __get__(self):
            return self.cache_config.decrement
        def __set__(self, double val):
            self.cache_config.decrement = val

    property apply_max_decrement:
        def __get__(self):
            return self.cache_config.apply_max_decrement
        def __set__(self, hbool_t val):
            self.cache_config.apply_max_decrement = val

    property max_decrement:
        def __get__(self):
            return self.cache_config.max_decrement
        def __set__(self, size_t val):
            self.cache_config.max_decrement = val

    property epochs_before_eviction:
        def __get__(self):
            return self.cache_config.epochs_before_eviction
        def __set__(self, int val):
            self.cache_config.epochs_before_eviction = val



    property apply_empty_reserve:
        def __get__(self):
            return self.cache_config.apply_empty_reserve
        def __set__(self, hbool_t val):
            self.cache_config.apply_empty_reserve = val


    property empty_reserve:
        def __get__(self):
            return self.cache_config.empty_reserve
        def __set__(self, double val):
            self.cache_config.empty_reserve = val

    # /* parallel configuration fields: */
    property dirty_bytes_threshold:
        def __get__(self):
            return self.cache_config.dirty_bytes_threshold
        def __set__(self, int val):
            self.cache_config.dirty_bytes_threshold = val
