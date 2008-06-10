from python cimport PyTuple_Check

cdef int require_tuple(object tpl, int none_allowed, int size, char* name) except -1:
    # Ensure that tpl is in fact a tuple, or None if none_allowed is nonzero.
    # If size >= 0, also ensure that the length matches.

    if (tpl is None and none_allowed) or \
      ( PyTuple_Check(tpl) and (size < 0 or len(tpl) == size)):
        return 1

    nmsg = ""
    smsg = ""
    if size >= 0:
        smsg = " of size %d" % size
    if none_allowed:
        nmsg = " or None"

    raise ValueError("%s must be a tuple%s%s." % (name, smsg, nmsg))
