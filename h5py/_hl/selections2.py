
from h5py import h5s

class ScalarReadSelection(object):

    def __init__(self, fspace, args):
        if args == ():
            self.mshape = None
        elif args == (Ellipsis,):
            self.mshape = ()
        else:
            raise ValueError("Illegal slicing argument for scalar dataspace")

        self.mspace = h5s.create(h5s.SCALAR)
        self.fspace = fspace

    def __iter__(self):
        self.mspace.select_all()
        yield self.fspace, self.mspace        

def select_read(fspace, args):
    
    if fspace.shape == ():
        return ScalarReadSelection(fspace, args)

    raise NotImplementedError()

