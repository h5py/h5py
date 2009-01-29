/***** Preamble block *********************************************************
* 
* This file is part of h5py, a low-level Python interface to the HDF5 library.
* 
* Copyright (C) 2008 Andrew Collette
* http://h5py.alfven.org
* License: BSD  (See LICENSE.txt for full license)
* 
* $Date$
* 
****** End preamble block ****************************************************/

/*
    Filter code is chosen in an ad-hoc manner to avoid conflict
    with PyTables LZO/BZIP2 implementation.
*/

#ifndef H5PY_LZF_H
#define H5PY_LZF_H

#define H5PY_FILTER_LZF 315

int register_lzf(void);

#endif

