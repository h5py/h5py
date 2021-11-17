ANACONDA_ORG="scipy-wheels-nightly";
pip install git+https://github.com/Anaconda-Server/anaconda-client;

if [[ "$TRAVIS_EVENT_TYPE" != "cron" && -z "$TRAVIS_TAG" ]] ; then
  echo "Not uploading wheels (build not for cron or git tag)"
  exit 0
fi

# rename wheels if not on a tag
# inserting commit count & hash from git describe
# e.g. h5py-3.3.0-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl to
#   h5py-3.3.0-3-g4320f2ea-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
if [[ -z "$TRAVIS_TAG" ]] ; then
  descr=$(git describe --tags)
  descr=${descr#*-}  # Chop off tag (should be version number)
  build_tag=${descr//-/_}  # Convert - to _ for build tag
  echo "Setting build tag to ${build_tag}"
  for whl in "${TRAVIS_BUILD_DIR}"/wheelhouse/h5py-*.whl; do
    newname=$(echo "$whl" | sed "s/\(h5py-[0-9][0-9]*[.[0-9]*]*-\)\(cp*\)/\1${build_tag}-\2/")
     if [ "$newname" != "$whl" ]; then
         mv "$whl" "$newname"
     fi
  done
fi

# upload wheels
if [[ -n "${ANACONDA_ORG_UPLOAD_TOKEN}" ]] ; then
   anaconda -t ${ANACONDA_ORG_UPLOAD_TOKEN} upload -u ${ANACONDA_ORG} "${TRAVIS_BUILD_DIR}"/wheelhouse/h5py-*.whl;
fi;
