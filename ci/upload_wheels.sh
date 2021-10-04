ANACONDA_ORG="test-upload";
pip install git+https://github.com/Anaconda-Server/anaconda-client;

# rename wheels
# appending timestamp to wheels package name
# e.g. h5py-3.3.0-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl to h5py-3.3.0-cp37-cp37m-20211004151033-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
for whl in ${TRAVIS_BUILD_DIR}/wheelhouse/h5py-*.whl; do 
	newname=$(echo "$whl" | sed "s/\(-[cp3(789)]*[m]*-\)\(manylinux*\)/\1$(date '+%Y%m%d%H%M%S')-\2/")
   if [ "$newname" != "$whl" ]; then
       mv $whl $newname
   fi
done

# upload wheels 
if [[ -n "${ANACONDA_ORG_UPLOAD_TOKEN}" ]] ; then
   anaconda -t ${ANACONDA_ORG_UPLOAD_TOKEN} upload --force -u ${ANACONDA_ORG} ${TRAVIS_BUILD_DIR}/wheelhouse/h5py-*.whl;
fi;

