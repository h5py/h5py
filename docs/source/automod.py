
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
        if what in ("function","method") and obj.__doc__.strip().startswith('('):
            lines = []
            for line in obj.__doc__.split("\n"):
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

    spx.connect('autodoc-process-signature', proc_sig)
    spx.connect('autodoc-process-docstring', proc_doc)

