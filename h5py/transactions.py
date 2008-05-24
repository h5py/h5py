import time

class TransactionError(StandardError):
    """ Base class for errors in the transactions module.
    """
    pass

class TransactionStateError(TransactionError):
    """ Attempted an operation which doesn't make sense in the current context.
    """
    pass

class IllegalTransactionError(TransactionError):
    pass


class Action(object):

    """ Represents the smallest component of a transaction.

        Each Action object is a simple state machine which can do three things:
        do, undo and commit.  Each of these actions is specified by a 3-tuple
        containing (callable, args, kwds).  Whatever the "do" callable does is
        assumed to be an atomic operation, which is fully reversed by the 
        "undo" operation.  The "commit" operation is optional, and can be
        used to e.g. clean up state information used for undo.

        States are
        READY:      The action has not been performed. 
                    Legal calls:
                        do() => DONE
        DONE:       The action has been performed.
                    Legal calls:
                        undo() => READY
                        commit() => COMMITTED
        COMMITTED:  The action has been performed and cannot be undone.
                    No legal calls.

        Peforming an illegal call (in the wrong state) will raise a 
        TransactionStateError.
    """

    READY = 0
    DONE = 1
    COMMITTED = 2

    def __init__(self, name, do, undo, commit=None):
        """ Create a new action object, from a name and a do/undo pair.

            "Do" and "undo" are required and must be 3-tuples containing
            (callable, args_tuple, keywords_dictionary).  "commit" is optional.

            do:     Perform some kind of atomic operation
            undo:   Completely reverse the effects of the "do" operation.
            commit: Clean up any temporary data created by the "do" operation.

            Any return value from the callable is ignored.  It must not raise
            an exception.

        """
        self.name = name
        self._validate(do, 'do')
        self._validate(undo, 'undo')
        if commit is not None:
            self._validate(commit, 'commit')
        self._do = do
        self._undo = undo
        self._commit = commit
        self.state = Action.READY

    def _validate(self, item, name):
        if isinstance(item, tuple)      and \
          len(item) == 3                and \
          hasattr(item[0],'__call__')   and \
          isinstance(item[1], tuple)    and \
          isinstance(item[2], dict):
            return
        raise ValueError('"%s" must be a 3-tuple (callable, args, kwds); got %s' % (name, str(item)) )

    def _call(self, tpl):
        if tpl is not None:
            f, a, k = tpl
            f(*a, **k)

    def do(self):
        if self.state != Action.READY:
            raise TransactionStateError('Action "%s" is not in the ready state (got %d)' % (self.name, self.state))
        self._call(self._do)
        self.state = Action.DONE

    def undo(self):
        if self.state != Action.DONE:
            raise TransactionStateError('Action "%s" is not in the "done" state for undo (got %d)' % (self.name, self.state))
        self._call(self._undo)
        self.state = Action.READY

    def commit(self):
        if self.state != Action.DONE:
            raise TransactionStateError('Action "%s" is not in the "done" state for commit (got %d)' % (self.name, self.state))
        self._call(self._commit)
        self.state = Action.COMMITTED


class TransactionManager(object):

    """ Provides locking and transaction support, via a stack of Action objects.
 
        Objects of this class are designed to manage access to a set of
        resources, manipulated by calling functions or bound methods.  It
        implements two conceptually different interfaces:

        1. Locking
        Since a single transaction manager can process commands affecting a
        variety of objects, you can "lock" an object by providing an identifing
        token (which could even be the object itself) via the function lock().
        Attempting to lock an object twice will raise ValueError.

        2. Transactions
        Transactions are implemented via a stack of Action objects.  Add actions
        to the manager via do().  Each action is immediately performed when
        this method is called.  At any time you can call commit(), to purge the
        stack, or rollback() to reverse all changes up to the last commit().
        Additionally, by calling undo() and redo(), you can manually walk the
        object through the entire stack of transaction states.

    """

    # Pointer states:
    # None:     uninitialized
    # 0:        off the bottom of the stack
    # <double>: index of an action on the stack

    def __init__(self, max_size=None):
        """ Create a new transaction manager.  The optional max_size keyword
            indicates the maximum allowed stack size.  When this limit is 
            reached, the oldest action is committed and discarded when a new
            action is added.
        """
        self.stack = {}
        self.locks = set()
        self.ptr = None
        if max_size is not None and max_size < 1:
            raise ValueError("Stack size must be at least 1 (got %d)" % max_size)
        self.max_size = max_size

    # --- Locking code --------------------------------------------------------

    def is_locked(self, item):
        """ Determine if this resource is currently locked.
        """
        return item in self.locks

    def lock(self, item):
        """ Lock a resource.  Raises ValueError if it's already locked.
        """
        if item in self.locks:
            raise ValueError('%s is already locked for transactions.' % str(item))
        self.locks.add(item)

    def unlock(self, item):
        """ Release a resource.  Raises ValueError if it's already locked.
        """
        if not item in self.locks:
            raise ValueError("%s is not locked for transactions." % item)
        self.locks.remove(item)

    # --- Transactions code ---------------------------------------------------

    def commit(self):
        """ Commit every action which is in the "done" state, and reset
            the stack.
        """ 
        for t in sorted(self.stack):
            action = self.stack[t]
            if action.state == Action.DONE:
                action.commit()
        self.stack = {}
        self.ptr = None
    
    def rollback(self):
        """ Undo every action which is in the "done" state, and reset
            the stack.
        """
        for t in sorted(self.stack, reverse=True):
            action = self.stack[t]
            if action.state == Action.DONE:
                action.undo()
        self.stack = {}
        self.ptr = None
        
    def can_redo(self):
        """ Determine if the stack/pointer system is capable of redoing the
            next action on the stack.  Fails if the pointer is None or at
            the top of the stack.
        """
        if self.ptr is None:
            return False
        assert self.ptr in self.stack or self.ptr == 0

        if self.ptr == max(self.stack):
            return False
        return True

    def can_undo(self):
        """ Determine if the stack/pointer system is capable of undoing the
            action currently pointed at.  Fails if the pointer is None, or
            off the bottom of the stack.
        """
        if self.ptr is None or self.ptr == 0:
            return False
        assert self.ptr in self.stack

        return True

    def do(self, action):
        """ Perform the given action and add it to the stack.  Implicitly
            discards any newer actions on the stack, and sets the pointer
            to the new item.  If the resulting stack size is greater than
            max_size, commit and discard the oldest action.

            The action's do() method is called before any modification is 
            made to the stack; if it raises an exception this won't trash
            the object.
        """
        action.do()

        assert len(self.stack) == 0 or self.ptr in self.stack

        for t in sorted(self.stack):
            if t > self.ptr:
                assert self.stack[t].state == Action.READY
                del self.stack[t]

        key = time.time()
        self.stack[key] = action
        self.ptr = key

        if self.max_size is not None and len(stack) > self.max_size:
            key = min(self.stack)
            self.stack[key].commit()
            del self.stack[key]

    def undo(self):
        """ Undo the action targeted by the current pointer, and move the
            pointer down one level on the stack.  Does nothing if the pointer 
            system isn't ready for an Undo.
        """
        if self.can_undo():
            self.stack[self.ptr].undo()
            keys = sorted(self.stack)
            idx = keys.index(self.ptr)
            if idx == 0:
                self.ptr = 0
            else:
                self.ptr = keys[idx-1]

    def redo(self):
        """ Increment the pointer and redo the resulting action.  Does nothing
            if the pointer system isn't ready for a Redo.
        """
        if self.can_redo():
            keys = sorted(self.stack)
            if self.ptr == 0:
                self.ptr = min(keys)
            else:
                idx = keys.index(self.ptr)
                self.ptr = keys[idx+1]
            self.stack[self.ptr].do()











