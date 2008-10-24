
"""
    Requires patched version of autodoc.py
    http://bugs.python.org/issue3422
"""
import re
from functools import partial


# --- Literal replacements for common class names ---

class_types = { "ObjectID": "h5py.h5.ObjectID",
                "GroupID": "h5py.h5g.GroupID",
                "FileID": "h5py.h5f.FileID",
                "DatasetID": "h5py.h5d.DatasetID",
                "TypeID": "h5py.h5t.TypeID",
                "[Dd]ataset creation property list": "h5py.h5p.PropDCID",
                "[Dd]ataset access property list": "h5py.h5p.PropDAID",
                "[Ff]ile creation property list": "h5py.h5p.PropFCID",
                "[Ff]ile access property list": "h5py.h5p.PropFAID"}


def mkclass(ipt, cls):
    return ":class:`%s <%s>`" % (ipt, cls)

replacements = {}
replacements.update((x, mkclass(x,y)) for x, y in class_types.items())

def replaceall(instring, rdict):
    for orig, new in rdict.iteritems():
        instring = instring.replace(orig, new)
    return instring


# --- "Smart" regexp replacements for UPPER_CASE constant names ---

# Just the constant name, with or without the h5x. prefix.
# Legal constant names are of the form CONST" or "h5p.CONST"
const_only = r"(?:h5[a-z]{0,2}\.)?[A-Z_][A-Z0-9_]+"

# Constant name embeddded in context (whitespace, parens, etc.)
const_pattern = r"(?:^|\s+)\W?%s\*?\W?\.?(?:$|\s+)" % const_only

const_category = r"%s\*" % const_only

# These match the regexp but are not valid constants
const_exclude = r"HDF5|API|H5|H5A|H5D|H5F|H5P|H5Z|" + \
                r"INT\s|UINT|STRING|LONG|PHIL|GIL|TUPLE|LIST|FORTRAN|" +\
                r"BOOL|NULL|\sNOT\s|\sSZIP\s"

def replace_constant(instring, mod, match):
    """ Callback for re.sub, to generate the ReST for a constant in-place """

    matchstring = instring[match.start():match.end()]

    if re.search(const_exclude, matchstring):
        return matchstring

    if re.search(const_category, matchstring):
        display = re.findall(const_category, matchstring)[0]
        target = display[0:-2]
        target = 'ref.'+target if 'h5' in target else 'ref.%s.%s' % (mod.split('.')[-1], target)
        rpl = ':ref:`%s <%s>`' % (display, target)
        #print rpl
        return re.sub(const_category, rpl, matchstring)
    else:  
        display = re.findall(const_only, matchstring)[0]
        target = display
        target = 'h5py.'+target if 'h5' in target else '%s.%s' % (mod, target)
        rpl = ':data:`%s <%s>`' % (display, target)
        return re.sub(const_only, rpl, matchstring)


# --- Resolve inline references to modules ---

mod_pattern = re.compile(r"\s+(?!h5py)(?P<name>h5[a-z]{1,2})\W?(?:$|\s+)"
    
module_pattern = r"\s+h5[a-z]{1,2}\W?(?:$|\s+)"
module_only = r"h5[a-z]{1,2}"
module_exclude = r"h5py"

def replace_module(instring, match):

    matchstring = instring[match.start():match.end()]

    if re.search(module_exclude, matchstring):
        return matchstring

    display = re.findall(module_only, matchstring)[0]
    target = display

    rpl = ':mod:`%s <%s>`' % (display, 'h5py.'+target)
    #print rpl
    return re.sub(module_only, rpl, matchstring)

# --- Resolve parameter lists

param_pattern = re.compile(r"^\s*\+\s+(?P<desc>[^\s\(].*[^\s\)])(?:\s+\((?P<default>[^\s\(].*[^\s\)])\))?$")

def replace_param(instring, match):

    desc, default = match.group('desc'), match.group('default')
    default = ' (%s) ' % default if default is not None else ''

    return ":param %s:%s" % (desc, default)

# --- Sphinx extension code ---


def setup(spx):

    def proc_doc(app, what, name, obj, options, lines):
        """ Process docstrings for modules and routines """

        final_lines = lines[:]

        # Remove the signature lines from the docstring
        if what in ("function", "method") and lines[0].strip().startswith('('):
            doclines = []
            arglines = []
            final_lines = arglines
            for line in lines:
                if len(line.strip()) == 0:
                    final_lines = doclines
                final_lines.append(line)

        # Resolve class names, constants and modules
        #print name
        if hasattr(obj, 'im_class'):
            mod = obj.im_class.__module__
        elif hasattr(obj, '__module__'):
            mod = obj.__module__
        else:
            mod = ".".join(name.split('.')[0:2])  # i.e. "h5py.h5z"
        lines[:] = [re.sub(const_pattern, partial(replace_constant, x, mod), x) for x in final_lines]
        lines[:] = [re.sub(module_pattern, partial(replace_module, x), x) for x in lines]
        lines[:] = [re.sub(param_pattern, partial(replace_param, x), x) for x in lines]
        lines[:] = [replaceall(x, replacements) for x in lines]
        #print "\n".join(lines)

    def proc_sig(app, what, name, obj, options, signature, return_annotation):
        """ Auto-generate function signatures from docstrings """

        def getsig(docstring):
            """ Get (sig, return) from a docstring, or None. """
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
            if len(sig) == 2: sig = sig[0]+" "+sig[1]  # autodoc hates "()"
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

