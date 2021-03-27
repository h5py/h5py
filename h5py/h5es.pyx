include "config.pxi"
from ._objects import phil, with_phil

@with_phil
def create(self):
    return H5EScreate()

@with_phil
def insert_request(self):
    cdef hid_t es_id
    cdef hid_t connector_id
    cdef void* request
    return H5ESinsert_request( es_id,  connector_id, request)