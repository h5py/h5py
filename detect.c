#include <stdio.h>
#include "hdf5.h"

int main(){
    fprintf(stdout, "%d.%d.%d\n", (int)H5_VERS_MAJOR, (int)H5_VERS_MINOR, (int)H5_VERS_RELEASE);
    return 0;
}
