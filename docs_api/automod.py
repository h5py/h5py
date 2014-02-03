
"""
    Requires patched version of autodoc.py
    http://bugs.python.org/issue3422
"""
import re
from functools import partial

# === Regexp replacement machinery ============================================

role_expr = re.compile(r"(:.+:(?:`.+`)?)")

def safe_replace(istr, expr, rpl):
    """ Perform a role-safe replacement of all occurances of "expr", using
        the callable "rpl".
    """
    outparts = []
    for part in role_expr.split(istr):
        if not role_expr.search(part):
            part = expr.sub(rpl, part)
        outparts.append(part)
    return "".join(outparts)


# === Replace literal class names =============================================

class_base = r"""
(?P<pre>
  \W+
)
(?P<name>%s)
(?P<post>
  \W+
)
"""

class_exprs = { "ObjectID": "h5py.h5.ObjectID",
                "GroupID": "h5py.h5g.GroupID",
                "FileID": "h5py.h5f.FileID",
                "DatasetID": "h5py.h5d.DatasetID",
                "TypeID": "h5py.h5t.TypeID",
                "[Dd]ataset creation property list": "h5py.h5p.PropDCID",
                "[Dd]ataset transfer property list": "h5py.h5p.PropDXID",
                "[Ff]ile creation property list": "h5py.h5p.PropFCID",
                "[Ff]ile access property list": "h5py.h5p.PropFAID",
                "[Ll]ink access property list": "h5py.h5p.PropLAID",
                "[Ll]ink creation property list": "h5py.h5p.PropLCID",
                "[Gg]roup creation property list": "h5py.h5p.PropGCID"}


class_exprs = dict( 
    (re.compile(class_base % x.replace(" ",r"\s"), re.VERBOSE), y) \
    for x, y in class_exprs.iteritems() )

def replace_class(istr):

    def rpl(target, match):
        pre, name, post = match.group('pre', 'name', 'post')
        return '%s:class:`%s <%s>`%s' % (pre, name, target, post)

    for expr, target in class_exprs.iteritems():
        rpl2 = partial(rpl, target)
        istr = safe_replace(istr, expr, rpl2)

    return istr

# === Replace constant and category expressions ===============================

# e.g. h5f.OBJ_ALL -> :data:`h5f.OBJ_ALL <h5py.h5f.OBJ_ALL>`
# and  h5f.OBJ*    -> :ref:`h5f.OBJ* <ref.h5f.OBJ>`

const_exclude = ['HDF5', 'API', 'H5', 'H5A', 'H5D', 'H5F', 'H5P', 'H5Z', 'INT',
                 'UINT', 'STRING', 'LONG', 'PHIL', 'GIL', 'TUPLE', 'LIST',
                 'FORTRAN', 'BOOL', 'NULL', 'NOT', 'SZIP']
const_exclude = ["%s(?:\W|$)" % x for x in const_exclude]
const_exclude = "|".join(const_exclude)

const_expr = re.compile(r"""
(?P<pre>
  (?:^|\s+)                   # Must be preceeded by whitespace or string start
  \W?                         # May have punctuation ( (CONST) or "CONST" )
  (?!%s)                      # Exclude known list of non-constant objects
)
(?P<module>h5[a-z]{0,2}\.)?   # Optional h5xx. prefix
(?P<name>[A-Z_][A-Z0-9_]+)    # The constant name itself
(?P<wild>\*)?                 # Wildcard indicates this is a category
(?P<post>
  \W?                         # May have trailing punctuation
  (?:$|\s+)                   # Must be followed by whitespace or end of string
)                      
""" % const_exclude, re.VERBOSE)

def replace_constant(istr, current_module):

    def rpl(match):
        mod, name, wild = match.group('module', 'name', 'wild')
        pre, post = match.group('pre', 'post')

        if mod is None:
            mod = current_module+'.'
            displayname = name
        else:
            displayname = mod+name

        if wild:
            target = 'ref.'+mod+name
            role = ':ref:'
            displayname += '*'
        else:
            target = 'h5py.'+mod+name
            role = ':data:'

        return '%s%s`%s <%s>`%s' % (pre, role, displayname, target, post)

    return safe_replace(istr, const_expr, rpl)


# === Replace literal references to modules ===================================

mod_expr = re.compile(r"""
(?P<pre>
  (?:^|\s+)                 # Must be preceeded by whitespace
  \W?                       # Optional opening paren/quote/whatever
)
(?!h5py)                    # Don't match the package name
(?P<name>h5[a-z]{0,2})      # Names of the form h5, h5a, h5fd
(?P<post>
  \W?                       # Optional closing paren/quote/whatever
  (?:$|\s+)                 # Must be followed by whitespace
)
""", re.VERBOSE)

def replace_module(istr):

    def rpl(match):
        pre, name, post = match.group('pre', 'name', 'post')
        return '%s:mod:`%s <h5py.%s>`%s' % (pre, name, name, post)

    return safe_replace(istr, mod_expr, rpl)


# === Replace parameter lists =================================================

# e.g. "    + STRING path ('/default')" -> ":param STRING path: ('/default')"

param_expr = re.compile(r"""
^
\s*
\+
\s+
(?P<desc>
  [^\s\(]
  .*
  [^\s\)]
)
(?:
  \s+
  \(
  (?P<default>
    [^\s\(]
    .*
    [^\s\)]
  )
  \)
)?
$
""", re.VERBOSE)

def replace_param(istr):
    """ Replace parameter lists.  Not role-safe. """

    def rpl(match):
        desc, default = match.group('desc', 'default')
        default = ' (%s) ' % default if default is not None else ''
        return ':param %s:%s' % (desc, default)

    return param_expr.sub(rpl, istr)



# === Begin Sphinx extension code =============================================

def is_callable(docstring):
    return str(docstring).strip().startswith('(')

def setup(spx):

    def proc_doc(app, what, name, obj, options, lines):
        """ Process docstrings for modules and routines """

        final_lines = lines[:]

        # Remove the signature lines from the docstring
        if is_callable(obj.__doc__):
            doclines = []
            arglines = []
            final_lines = arglines
            for line in lines:
                if len(line.strip()) == 0:
                    final_lines = doclines
                final_lines.append(line)

        # Resolve class names, constants and modules
        if hasattr(obj, 'im_class'):
            mod = obj.im_class.__module__
        elif hasattr(obj, '__module__'):
            mod = obj.__module__
        else:
            mod = ".".join(name.split('.')[0:2])  # i.e. "h5py.h5z"
        mod = mod.split('.')[1]  # i.e. 'h5z'

        del lines[:]
        for line in final_lines:
            #line = replace_param(line)
            line = replace_constant(line, mod)
            line = replace_module(line)
            line = replace_class(line)
            line = line.replace('**kwds', '\*\*kwds').replace('*args','\*args')
            lines.append(line)




    def proc_sig(app, what, name, obj, options, signature, return_annotation):
        """ Auto-generate function signatures from docstrings """

        def getsig(docstring):
            """ Get (sig, return) from a docstring, or None. """
            if not is_callable(docstring):
                return None

            lines = []
            for line in docstring.split("\n"):
                if len(line.strip()) == 0:
                    break
                lines.append(line)
            rawsig = " ".join(x.strip() for x in lines)

            if '=>' in rawsig:
                sig, ret = tuple(x.strip() for x in rawsig.split('=>'))
            elif '->' in rawsig:
                sig, ret = tuple(x.strip() for x in rawsig.split('->'))
            else:
                sig = rawsig
                ret = None

            if sig == "()":
                sig = "( )" # Why? Ask autodoc.

            return (sig, ret)

        sigtuple = getsig(obj.__doc__)

        return sigtuple

    spx.connect('autodoc-process-signature', proc_sig)
    spx.connect('autodoc-process-docstring', proc_doc)

