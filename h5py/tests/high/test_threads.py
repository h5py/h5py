import sys
import threading
import h5py

from h5py import tests

class TestThreads(tests.HTest):
    
    def test_exc(self):
        """ (Threads) Exception support in non-native threads """

        results = {}

        def badfunc():

            h5py.h5e.unregister_thread(h5py.h5e.NullErrorHandler)
            try:
                h5py.h5f.is_hdf5('missing')
            except Exception:
                results['badfunc'] = False
            else:
                results['badfunc'] = True

        def goodfunc():
            h5py.h5e.register_thread()
            try:
                h5py.h5f.is_hdf5('missing')
            except IOError:
                results['goodfunc'] = True
            else:
                results['goodfunc'] = False

        t = threading.Thread(target=badfunc)
        t.start()
        t.join()
        t = threading.Thread(target=goodfunc)
        t.start()
        t.join()
        self.assert_(results['goodfunc'])
        self.assert_(results['badfunc'])

        
        
