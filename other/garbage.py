
"""
    Demonstrates garbage messages printed to stderr for containership
    testing, when performed in new threads.
"""

from threading import Thread

import h5py

def demonstrate():
    with h5py.File('foo', 'w', driver='core') as f:
        print('x' in f)

if __name__ == '__main__':
    print("Main thread")
    demonstrate()
    thread = Thread(target=demonstrate)
    print("New thread")
    thread.start()
    thread.join()
