
"""
    Example demonstrating some features of groups in HDF5, and how to
    use them from h5py.  
"""

import h5py

groups = ['/foo', '/foo/bar', '/foo/bar/baz',
          '/grp1', '/grp1/grp2', '/mygroup']

f = h5py.File('group_test.hdf5','w')

for grpname in groups:
    f.create_group(grpname)

print "Root group names:"

for name in f:
    print "   ", name

print "Root group info:"

for name, grp in f.iteritems():
    print "    %s: %s items" % (name, len(grp))

if h5py.version.api_version_tuple >= (1,8):

    def treewalker(name):
        """ Callback function for visit() """
        print "    Called for %s" % name

    print "Walking..."
    f.visit(treewalker)

    print "Copying /foo to /mygroup/newfoo..."
    f.copy("/foo", "/mygroup/newfoo")

    print "Walking again..."
    f.visit(treewalker)

    g = f['/grp1']

    print "Walking from /grp1..."
    g.visit(treewalker)

else:
    print "HDF5 1.8 is needed for extra demos"

f.close()
