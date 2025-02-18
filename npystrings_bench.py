# DNM
import argparse
import random
import string
import time

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-width", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=100)
    parser.add_argument("--num-strings", type=int, default=1_000_000)
    parser.add_argument("--dtype", type=str, default="T")
    args = parser.parse_args()

    data = [
        "".join(
            random.choices(
                string.ascii_letters,
                k=random.randint(args.min_width, args.max_width),
            )
        )
        for _ in range(args.num_strings)
    ]
    data = np.asarray(data, dtype=args.dtype)
    print(f"write dtype: {data.dtype}")

    with h5py.File("bench.h5", "w") as f:
        t0 = time.time()
        f.create_dataset("data", data=data, dtype=h5py.string_dtype())
        t1 = time.time()
    print(f"write: {t1 - t0:.3f}s")
    with h5py.File("bench.h5", "r") as f:
        t0 = time.time()
        ds = f["data"]
        print(f"dset dtype: {ds.dtype}")
        data2 = ds.asstr()[:] if args.dtype == "O" else ds[:]
        t1 = time.time()
    print(f"read dtype: {data2.dtype}")
    print(f"read: {t1 - t0:.3f}s")

    np.testing.assert_array_equal(data2, data)


if __name__ == "__main__":
    main()
