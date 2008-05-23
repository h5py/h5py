import time

class TransactionError(StandardError):
    """ Base class for errors in the transactions module.
    """
    pass

class TransactionStateError(TransactionError):
    """ Illegal state change inside an Action object.
    """
    pass

class NotTransactableError(TransactionError):
    """ A non-transactable operation was attempted while in transaction mode.
    """
    pass

class DeadTransactionError(TransactionError):
    """ A transactable operation was attempted while not in transaction mode.
    """
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

    active = property(lambda self: self.stack is not None)

    def __init__(self, max_size=None):
        """ Create a new transaction manager, inactive by default.
        """
        self.stack = None
        self.ptr = None
        if max_size < 1:
            raise ValueError("Stack size must be at least 1 (got %d)" % max_size)
        self.max_size = max_size

    def _check(self):
        if not self.active:
            raise DeadTransactionError("No current transaction.")

    def begin(self):
        """ Begin a new transaction.  Implicitly commits any pending actions.
        """
        if self.active:
            self.commit()
        self.stack = {}
        self.ptr = None

    def commit(self):
        """ Commit every action which is in the "done" state, and destroy
            the stack.
        """ 
        self._check()
        for t in sorted(self.stack):
            action = self.stack[t]
            if action.state == Action.DONE:
                action.commit()
        self.stack = None
        self.ptr = None
    
    def rollback(self):
        """ Undo every action which is in the "done" state, and destroy
            the stack.
        """
        self._check()
        for t in sorted(self.stack, reverse=True):
            action = self.stack[t]
            if action.state == Action.DONE:
                action.undo()
        self.stack = None
        self.ptr = None
        
    def at_top(self):
        """ Check if the pointer is at the top of the stack.
        """
        self._check()
        if len(self.stack) == 0 or self.ptr == max(self.stack):
            return True
        return False

    def at_bottom(self):
        """ Check if the pointer is at the bottom of the stack.
        """
        self._check()
        if len(self.stack) == 0 or self.ptr == min(self.stack):
            return True
        return False

    def do(self, action):
        """ Perform the given action and add it to the stack.  Implicitly
            discards any newer actions on the stack, and sets the pointer
            to the new item.  If the resulting stack size is greater than
            max_size, commit and discard the oldest action.

            The action's do() method is called before any modification is 
            made to the stack.
        """
        self._check()
        action.do()

        if self.ptr is not None:
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
            is None or off the bottom of the stack.
        """
        self._check()
        if self.ptr is not None and self.ptr >= 0:
            self.stack[self.ptr].undo()
            keys = sorted(self.stack)
            idx = keys.index(self.ptr)
            if self.at_bottom():
                self.ptr = -1
            else:
                self.ptr = keys[idx-1]

    def redo(self):
        """ Increment the pointer and redo the resulting action.  Does nothing
            if the pointer is already at the top of the stack, or is None.
        """
        self._check()
        if not self.at_top() and self.ptr is not None:
            keys = sorted(self.stack)
            if self.ptr < 0:
                self.ptr = min(keys)
            else:
                idx = keys.index(self.ptr)
                self.ptr = keys[idx+1]
            self.stack[self.ptr].do()



class LockManger(object):

    """
        Trivial locking class.
    """

    def __init__(self):
        self.locks = set()

    def is_locked(self, item):
        return item in self.locks

    def lock(self, item):

        if item in self.locks:
            return False
        self.locks.add(item)
        return True

    def release(self, item):
        
        if not item in self.locks:
            raise ValueError("%s is not locked" % item)
        self.locks.remove(item)



























