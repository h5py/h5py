
/* Generate an HDF5 test file for attributes unit test. */

#include "hdf5.h"

int attributes(char* filename){
    
    char val1[] = "This is a string.";
    int val2 = 42;
    hsize_t val3_dims = 4;
    int val3[4];
    int val4 = -34;
    
    val3[0] = 0;
    val3[1] = 1;
    val3[2] = 2;
    val3[3] = 3;

    hid_t sid_scalar=0;
    hid_t sid_array=0;
    hid_t fid=0;
    hid_t gid=0;
    hid_t string_id=0;
    hid_t aid=0;

    int retval=1;

    sid_scalar = H5Screate(H5S_SCALAR);
    if(sid_scalar<0) goto out;

    sid_array = H5Screate_simple(1, &val3_dims, NULL);
    if(sid_array<0) goto out;

    string_id = H5Tcopy(H5T_C_S1);
    if(string_id<0) goto out;

    if(H5Tset_size(string_id, 18)<0) goto out; /* string is 17 chars, plus NULL */

    fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if(fid<0) goto out;

    gid = H5Gcreate(fid, "Group", -1);
    if(gid<0) goto out;

    /* 1: "String attribute", exactly 18 bytes with null terminator, scalar,
          value "This is a string." */
    aid = H5Acreate(gid, "String Attribute", string_id, sid_scalar, H5P_DEFAULT);
    if(aid<0) goto out;
    if(H5Awrite(aid, string_id, &val1) < 0) goto out;
    if(H5Aclose(aid)<0) goto out;

    /* 2: "Integer", 32-bit little-endian, scalar, value 42. */
    aid = H5Acreate(gid, "Integer", H5T_STD_I32LE, sid_scalar, H5P_DEFAULT);
    if(aid<0) goto out;
    if(H5Awrite(aid, H5T_NATIVE_INT, &val2)<0) goto out;
    if(H5Aclose(aid)<0) goto out;

    /* 3: "Integer array", 4-element 32-bit little endian, value [0,1,2,3] */
    aid = H5Acreate(gid, "Integer Array", H5T_STD_I32LE, sid_array, H5P_DEFAULT);
    if(aid<0) goto out;
    if(H5Awrite(aid, H5T_NATIVE_INT, &val3)<0) goto out;
    if(H5Aclose(aid)<0) goto out;

    /* 4: "Byte", 8-bit "little-endian" integer, value -34 */
    aid = H5Acreate(gid, "Byte", H5T_STD_I8LE, sid_scalar, H5P_DEFAULT);
    if(aid<0) goto out;
    if(H5Awrite(aid, H5T_NATIVE_INT, &val4)<0) goto out;
    if(H5Aclose(aid)<0) goto out;

    retval = 0;  /* got here == success */

    out:

    if(sid_scalar) H5Sclose(sid_scalar);
    if(sid_array)  H5Sclose(sid_array);
    if(fid) H5Fclose(fid);
    if(gid) H5Gclose(gid);
    if(string_id) H5Tclose(string_id);

    return retval;
}

int main(int argc, char **argv){

    if(argc!=2) return 2;
    return attributes(argv[1]);
}



    
