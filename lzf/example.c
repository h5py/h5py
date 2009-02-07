/*
    Copyright (C) 2009 Andrew Collette
    http://h5py.alfven.org
    License: BSD (see LICENSE.txt)

    Example program demonstrating use of the LZF filter from C code.

    The LZF filter provides high-speed compression with acceptable compression
    performance, resulting in much faster performance than DEFLATE, at the
    cost of a slightly worse compression ratio. It's appropriate for large
    datasets of low to moderate complexity, for which some compression is
    much better than none, but for which the speed of DEFLATE is unacceptable.

    It's recommended to use the SHUFFLE filter with LZF, as it's virtually
    free, supported by all current versions of HDF5, and can significantly
    improve the compression ratio.

    The filter is completely stateless, and so is safe to statically
    link into the final program if desired. Using gcc with option -O1
    or higher is recommended.  If building with HDF5 1.8, you must
    define H5_USE_16_API when compiling.

    To compile this program:

    h5cc -DH5_USE_16_API lzf/*.c lzf_filter.c example.c -o example

    To run:

    $ ./example
    $ h5ls -v test_lzf.hdf5 
    Opened "test_lzf.hdf5" with sec2 driver.
    dset                     Dataset {100/100, 100/100, 100/100}
        Location:  0:1:0:976
        Links:     1
        Modified:  2009-01-28 21:51:20 PST
        Chunks:    {1, 100, 100} 40000 bytes
        Storage:   4000000 logical bytes, 529745 allocated bytes, 755.08% utilization
        Filter-0:  shuffle-2 OPT {4}
        Filter-1:  lzf-32000 OPT {1, 261, 0}
        Type:      native float
*/

#include <stdio.h>
#include "hdf5.h"
#include "lzf_filter.h"

#define SIZE 100*100*100
#define SHAPE {100,100,100}
#define CHUNKSHAPE {1,100,100}

int main(){

    static float data[SIZE];
    static float data_out[SIZE];
    const hsize_t shape[] = SHAPE;
    const hsize_t chunkshape[] = CHUNKSHAPE;
    int r, i;
    int return_code = 1;

    hid_t fid, sid, dset, plist = 0;

    for(i=0; i<SIZE; i++){
        data[i] = i;
    }

    /* Register the filter with the library */
    r = register_lzf();
    if(r<0) goto failed;

    sid = H5Screate_simple(3, &shape, NULL);
    if(sid<0) goto failed;

    fid = H5Fcreate("test_lzf.hdf5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if(fid<0) goto failed;

    plist = H5Pcreate(H5P_DATASET_CREATE);
    if(plist<0) goto failed;

    /* Chunked layout required for filters */
    r = H5Pset_chunk(plist, 3, &chunkshape);
    if(r<0) goto failed;

    /* Use of the shuffle filter VASTLY improves performance of this
       and other block-oriented compression filters.  Be sure to add
       this before the compression filter!
    */
    r = H5Pset_shuffle(plist);
    if(r<0) goto failed;

    /* Note the "optional" flag is necessary, as with the DEFLATE filter */
    r = H5Pset_filter(plist, H5PY_FILTER_LZF, H5Z_FLAG_OPTIONAL, 0, NULL);
    if(r<0) goto failed;

    dset = H5Dcreate(fid, "dset", H5T_NATIVE_FLOAT, sid, plist);
    if(dset<0) goto failed;
    
    r = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
    if(r<0) goto failed;

    r = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data_out);
    if(r<0) goto failed;

    for(i=0;i<SIZE;i++){
        if(data[i] != data_out[i]) goto failed;
    }

    fprintf(stdout, "Success!\n");

    return_code = 0;

    failed:

    if(dset>0)  H5Dclose(dset);
    if(sid>0)   H5Sclose(sid);
    if(plist>0) H5Pclose(plist);
    if(fid>0)   H5Fclose(fid);

    return return_code;
}

