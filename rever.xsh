$PROJECT = $GITHUB_ORG = $GITHUB_REPO = 'h5py'

$ACTIVITIES = ['authors', 'version_bump', 'changelog', 'tag', 'push_tag', 'pypi',
               'ghrelease', 'conda_forge']

$AUTHORS_FILENAME = 'AUTHORS.rst'

def version_tuple(version):
    parts = version.split('.')
    major, minor = parts[:2]
    bugfix = parts[1] if len(parts) > 2 else "0"
    vt = "version_tuple = _H5PY_VERSION_CLS({0}, {1}, {2}, None, None, None)"
    return vt.format(major, minor, bugfix)


$VERSION_BUMP_PATTERNS = [
    ('setup.py', r'VERSION\s*= .*', "VERSION = '$VERSION'"),
    ('docs/conf.py', r'release\s*=.*', "release = '$VERSION'"),
    ('h5py/version', r'version_tuple\s*=\s*_H5PY_VERSION_CLS.*', version_tuple)
    ]
$CHANGELOG_FILENAME = 'CHANGELOG.rst'
$CHANGELOG_TEMPLATE = 'TEMPLATE.rst'

$CONDA_FORGE_SOURCE_URL = ('https://github.com/h5py/h5py/releases/'
                           'download/$VERSION/h5py-$VERSION.tar.gz')
