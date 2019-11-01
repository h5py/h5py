"""
Author: Daniel Berke, berke.daniel@gmail.com
Date: October 27, 2019
Requirements: h5py>=2.10.0, unyt>=v2.4.0
Notes: This short example script shows how to save unit information attached
to a `unyt_array` using `attrs` in HDF5, and recover it upon reading the file.
It uses the Unyt package (https://github.com/yt-project/unyt) because that's
what I'm familiar with, but presumably similar options exist for Pint and
astropy.units.
"""

import h5py
import tempfile
import unyt as u

# Set up a temporary file for this example.
tf = tempfile.TemporaryFile()
f = h5py.File(tf, 'a')

# Create some mock data with moderately complicated units (this is the
# dimensional representation of Joules of energy).
test_data = [1, 2, 3, 4, 5] * u.kg * ( u.m / u.s ) ** 2
print(test_data.units)
# kg*m**2/s**2

# Create a data set to hold the numerical information:
f.create_dataset('stored data', data=test_data)

# Save the units information as a string in `attrs`.
f['stored data'].attrs['units'] = str(test_data.units)

# Now recover the data, using the saved units information to reconstruct the
# original quantities.
reconstituted_data = u.unyt_array(f['stored data'],
                                  units=f['stored data'].attrs['units'])

print(reconstituted_data.units)
# kg*m**2/s**2

assert reconstituted_data.units == test_data.units
