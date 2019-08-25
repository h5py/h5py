import re

$PROJECT = $GITHUB_ORG = $GITHUB_REPO = 'h5py'
$ACTIVITIES = ['authors', 'version_bump', 'changelog',
               'tag', 'push_tag', 'ghrelease', 'pypi',
              ]

version_re = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<bugfix>\d+)"
                        r"(\.(?P<pre>(?!post|dev)[^.]+))?"
                        r"(.post(?P<post>\d+))?(.dev(?P<dev>\d+))?")

def replace_version_tuple(ver):
    parts = version_re.match(ver).groupdict()
    ver_tup = ("version_tuple = _H5PY_VERSION_CLS({major}, {minor}, {bugfix}, "
               "{pre!r}, {post}, {dev})")
    return ver_tup.format(**parts)


$VERSION_BUMP_PATTERNS = [
    ('setup.py', r'VERSION\s*=.*', "VERSION = '$VERSION'")
    ('docs/conf.py', r'release\s*=.*', "release = '$VERSION'"),
    ('h5py/version.py', r'version_tuple\s*=.*', replace_version_tuple),
    ]
$CHANGELOG_FILENAME = 'CHANGELOG.rst'
$CHANGELOG_TEMPLATE = 'TEMPLATE.rst'
