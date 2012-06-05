
"""
    Implements ObjectID base class and global object registry.

    It used to be that we could store the HDF5 identifier in an ObjectID
    and simply close it when the object was deallocated.  However, since
    HDF5 1.8.5 they have started recycling object identifiers, which
    breaks this system.

    We now use a global registry of object identifiers.  This is implemented
    via a dictionary which maps an integer representation of the identifier
    to a weak reference of an ObjectID.  There is only one ObjectID instance
    in the universe for each integer identifier.  When the HDF5 reference
    count for a identifier reaches zero, HDF5 closes the object and reclaims
    the identifier. When this occurs, the identifier and weak reference must
    be deleted from the registry. If an ObjectID is deallocated, it is deleted
    from the registry and the HDF5 reference count is decreased, HDF5 closes
    and reclaims the identifier for future use.

    All interactions with the registry must be synchronized for thread safety.
    You must acquire "registry.lock" before interacting with the registry. The
    registry is not internally synchronized, in the interest of performance: we
    don't want the same thread attempting to acquire the lock multiple times
    during a single operation, if we can avoid it.

    All ObjectIDs and subclasses thereof should be opened with the "open"
    classmethod factory function, such that an existing ObjectID instance can
    be returned from the registry when appropriate.
"""

from defs cimport *

from weakref import KeyedRef, ref

## {{{ http://code.activestate.com/recipes/577336/ (r3)
from cpython cimport pythread
from cpython.exc cimport PyErr_NoMemory

cdef class FastRLock:
    """Fast, re-entrant locking.

    Under uncongested conditions, the lock is never acquired but only
    counted.  Only when a second thread comes in and notices that the
    lock is needed, it acquires the lock and notifies the first thread
    to release it when it's done.  This is all made possible by the
    wonderful GIL.
    """
    cdef pythread.PyThread_type_lock _real_lock
    cdef long _owner            # ID of thread owning the lock
    cdef int _count             # re-entry count
    cdef int _pending_requests  # number of pending requests for real lock
    cdef bint _is_locked        # whether the real lock is acquired

    def __cinit__(self):
        self._owner = -1
        self._count = 0
        self._is_locked = False
        self._pending_requests = 0
        self._real_lock = pythread.PyThread_allocate_lock()
        if self._real_lock is NULL:
            PyErr_NoMemory()

    def __dealloc__(self):
        if self._real_lock is not NULL:
            pythread.PyThread_free_lock(self._real_lock)
            self._real_lock = NULL

    def acquire(self, bint blocking=True):
        return lock_lock(self, pythread.PyThread_get_thread_ident(), blocking)

    def release(self):
        if self._owner != pythread.PyThread_get_thread_ident():
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    # compatibility with threading.RLock

    def __enter__(self):
        # self.acquire()
        return lock_lock(self, pythread.PyThread_get_thread_ident(), True)

    def __exit__(self, t, v, tb):
        # self.release()
        if self._owner != pythread.PyThread_get_thread_ident():
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    def _is_owned(self):
        return self._owner == pythread.PyThread_get_thread_ident()


cdef inline bint lock_lock(FastRLock lock, long current_thread, bint blocking) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if lock._count:
        # locked! - by myself?
        if current_thread == lock._owner:
            lock._count += 1
            return 1
    elif not lock._pending_requests:
        # not locked, not requested - go!
        lock._owner = current_thread
        lock._count = 1
        return 1
    # need to get the real lock
    return _acquire_lock(
        lock, current_thread,
        pythread.WAIT_LOCK if blocking else pythread.NOWAIT_LOCK)

cdef bint _acquire_lock(FastRLock lock, long current_thread, int wait) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if not lock._is_locked and not lock._pending_requests:
        # someone owns it but didn't acquire the real lock - do that
        # now and tell the owner to release it when done. Note that we
        # do not release the GIL here as we must absolutely be the one
        # who acquires the lock now.
        if not pythread.PyThread_acquire_lock(lock._real_lock, wait):
            return 0
        #assert not lock._is_locked
        lock._is_locked = True
    lock._pending_requests += 1
    with nogil:
        # wait for the lock owning thread to release it
        locked = pythread.PyThread_acquire_lock(lock._real_lock, wait)
    lock._pending_requests -= 1
    #assert not lock._is_locked
    #assert lock._count == 0
    if not locked:
        return 0
    lock._is_locked = True
    lock._owner = current_thread
    lock._count = 1
    return 1

cdef inline void unlock_lock(FastRLock lock) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    #assert lock._owner == pythread.PyThread_get_thread_ident()
    #assert lock._count > 0
    lock._count -= 1
    if lock._count == 0:
        lock._owner = -1
        if lock._is_locked:
            pythread.PyThread_release_lock(lock._real_lock)
            lock._is_locked = False
## end of http://code.activestate.com/recipes/577336/ }}}


cdef class _Registry:

    cdef object _data
    cdef readonly FastRLock lock

    def __cinit__(self):
        self._data = {}
        self.lock = FastRLock()

    __hash__ = None # Avoid Py3 warning

    def cleanup(self):
        "Manage invalid identifiers"
        deadlist = []
        for key in self._data:
            val = self._data[key]
            val = val()
            if val is None:
                deadlist.append(key)
                continue
            if not val.valid:
                deadlist.append(key)
        for key in deadlist:
            del self._data[key]

    def __getitem__(self, key):
        o = self._data[key]()
        if o is None:
            # This would occur if we had open objects and closed their
            # file, causing the objects identifiers to be reclaimed.
            # Now we clean up the registry when we close a file (or any
            # other identifier, for that matter), so in practice this
            # condition never obtains.
            del self._data[key]
            # We need to raise a KeyError:
            o = self._data[key]()
        return o

    def __setitem__(self, key, val):
        # this method should only be called by ObjectID.open
        self._data[key] = ref(val)

    def __delitem__(self, key):
        # we need to synchronize removal of the id from the
        # registry with decreasing the HDF5 reference count:
        try:
            del self._data[key]
        except KeyError:
            pass
        try:
            H5Idec_ref(key)
        except RuntimeError:
            # dec_ref failed because object was explicitly closed
            pass


registry = _Registry()


cdef class ObjectID:

    """
        Represents an HDF5 identifier.

    """

    property fileno:
        def __get__(self):
            cdef H5G_stat_t stat
            H5Gget_objinfo(self.id, '.', 0, &stat)
            return (stat.fileno[0], stat.fileno[1])

    property valid:
        def __get__(self):
            if not self.id:
                return False
            res = H5Iget_type(self.id) > 0
            if not res:
                self.id = 0
            return res

    def __cinit__(self, id):
        self.id = id
        self.locked = 0
        with registry.lock:
            registry[id] = self

    def __dealloc__(self):
        if not self.locked:
            try:
                with registry.lock:
                    del registry[self.id]
            except AttributeError:
                # library being torn down, registry is None
                pass

    def __nonzero__(self):
        return self.valid

    def __copy__(self):
        cdef ObjectID cpy
        cpy = type(self)(self.id)
        return cpy

    def __richcmp__(self, object other, int how):
        """ Default comparison mechanism for HDF5 objects (equal/not-equal)

        Default equality testing:
        1. Objects which are not both ObjectIDs are unequal
        2. Objects with the same HDF5 ID number are always equal
        3. Objects which hash the same are equal
        """
        cdef bint equal = 0

        if how != 2 and how != 3:
            return NotImplemented

        if isinstance(other, ObjectID):
            if self.id == other.id:
                equal = 1
            else:
                try:
                    equal = hash(self) == hash(other)
                except TypeError:
                    pass

        if how == 2:
            return equal
        return not equal

    def __hash__(self):
        """ Default hashing mechanism for HDF5 objects

        Default hashing strategy:
        1. Try to hash based on the object's fileno and objno records
        2. If (1) succeeds, cache the resulting value
        3. If (1) fails, raise TypeError
        """
        cdef H5G_stat_t stat

        if self._hash is None:
            try:
                H5Gget_objinfo(self.id, '.', 0, &stat)
                self._hash = hash((stat.fileno[0], stat.fileno[1], stat.objno[0], stat.objno[1]))
            except Exception:
                raise TypeError("Objects of class %s cannot be hashed" % self.__class__.__name__)

        return self._hash

    @classmethod
    def open(cls, id):
        """ Return a representation of an HDF5 identifier """
        with registry.lock:
            try:
                res = registry[id]
            except KeyError:
                res = cls(id)
            return res


cdef hid_t pdefault(ObjectID pid):

    if pid is None:
        return <hid_t>H5P_DEFAULT
    return pid.id
