/*
    Copyright (C) 2009 Andrew Collette
    http://h5py.alfven.org
    License: BSD (see LICENSE.txt)

    Example program demonstrating use of the LZF filter from C code.

    To compile this program:

    h5cc -DH5_USE_16_API lzf/*.c lzf_filter.c example.c -o example

    To run:

    $ ./example
    Success!
    $ h5ls -v test_lzf.hdf5 
    Opened "test_lzf.hdf5" with sec2 driver.
    dset                     Dataset {100/100, 100/100, 100/100}
        Location:  0:1:0:976
        Links:     1
        Modified:  2009-02-15 16:35:11 PST
        Chunks:    {1, 100, 100} 40000 bytes
        Storage:   4000000 logical bytes, 174288 allocated bytes, 2295.05% utilization
        Filter-0:  shuffle-2 OPT {4}
        Filter-1:  lzf-32000 OPT {1, 261, 40000}
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

    sid = H5Screate_simple(3, shape, NULL);
    if(sid<0) goto failed;

    fid = H5Fcreate("test_lzf.hdf5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if(fid<0) goto failed;

    plist = H5Pcreate(H5P_DATASET_CREATE);
    if(plist<0) goto failed;

    /* Chunked layout required for filters */
    r = H5Pset_chunk(plist, 3, chunkshape);
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

