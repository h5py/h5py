#!/usr/bin/env python
# Install dependencies from a "[metadata] setup-requires = ..." section in
# setup.cfg, then run real-setup.py.
# From https://bitbucket.org/dholth/setup-requires

import sys, os, subprocess, codecs, pkg_resources

sys.path[0:0] = ['setup-requires']
pkg_resources.working_set.add_entry('setup-requires')

try:
    import configparser
except:
    import ConfigParser as configparser

def get_requirements():
    if not os.path.exists('setup.cfg'): return
    config = configparser.ConfigParser()
    config.readfp(codecs.open('setup.cfg', encoding='utf-8'))
    setup_requires = config.get('metadata', 'setup-requires')
    specifiers = [line.strip() for line in setup_requires.splitlines()]
    for specifier in specifiers:
        try:
            pkg_resources.require(specifier)
        except pkg_resources.DistributionNotFound:
            yield specifier

try:
    to_install = list(get_requirements())
    if to_install:
        subprocess.call([sys.executable, "-m", "pip", "install", 
            "-t", "setup-requires"] + to_install)
except (configparser.NoSectionError, configparser.NoOptionError):
    pass

# Run real-setup.py
exec(compile(open("real-setup.py").read().replace('\\r\\n', '\\n'),
    __file__,
    'exec'))

