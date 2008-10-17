
"""
    Requires patched version of autodoc.py
    http://bugs.python.org/issue3422
"""

def setup(spx):

    def proc_doc(app, what, name, obj, options, lines):
        if what in ("function", "method") and lines[0].strip().startswith('('):
            doclines = []
            arglines = []
            final_lines = arglines
            for line in lines:
                if len(line.strip()) == 0:
                    final_lines = doclines
                final_lines.append(line)
            lines[:] = final_lines

    def proc_sig(app, what, name, obj, options, signature, return_annotation):

        def getsig(docstring):
            """ Get sig, return from a docstring, or None. """
            if docstring is None or not docstring.strip().startswith('('):
                return None

            lines = []
            for line in docstring.split("\n"):
                if len(line.strip()) == 0:
                    break
                lines.append(line)
            sig = [x.strip() for x in " ".join(lines).split('=>')]
            ret = ''
            if len(sig) == 2:
                sig, ret = sig
                ret = " -> "+ret
            else:
                sig = sig[0]
            if len(sig) == 2: sig = sig[0]+" "+sig[1]  # stupid bug in autodoc
            return (sig, ret)

        if what not in ("function", "method"):
            return None

        sigtuple = getsig(obj.__doc__)

        # If it's a built-in fuction it MUST have a docstring or
        # Sphinx shits a giant red box into the HTML
        if sigtuple is None and \
          (not (hasattr(obj, 'im_func') or hasattr(obj, 'func_code'))):
            return ('(...)', '')

        return sigtuple

    spx.connect('autodoc-process-signature', proc_sig)
    spx.connect('autodoc-process-docstring', proc_doc)

