New features
------------

* This prevents seg-fault at exit where Python code may be called after
  the interpreter is already half way destroyed, usually in threaded environment.

Development
-----------

* This allows profiling and code coverage tools to work also
  on cython code whih is great for developers.
