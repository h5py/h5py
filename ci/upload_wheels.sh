ANACONDA_ORG="scipy-wheels-nightly";
pip install git+https://github.com/Anaconda-Server/anaconda-client;

if [[ "$TRAVIS_EVENT_TYPE" != "cron" && -z "$TRAVIS_TAG" ]] ; then
  echo "Not uploading wheels (build not for cron or git tag)"
  exit 0
fi

# rename wheels if not on a tag
# inserting short commit hash as build tag in wheel filename
# e.g. h5py-3.3.0-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl to
#   h5py-3.3.0-0e2d161a0-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
if [[ -z "$TRAVIS_TAG" ]] ; then
  # build tag has to start with a decimal digit, so prefix 0 on commit hash
  build_tag="0$(git rev-parse --short=8 HEAD)"
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
