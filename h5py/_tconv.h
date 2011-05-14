#ifndef TCONV_H
#define TCONV_H

#include "Python.h"
#include "hdf5.h"

typedef int (*conv_operator_t)(void* ipt, void* opt, void* bkg, void* priv)
typedef herr_t (*init_operator_t)(hid_t src, hid_t dst, void** priv)

/* Obtain a reference to the datatype representing an in-memory Python object */
hid_t get_python_obj(void)


#endif
