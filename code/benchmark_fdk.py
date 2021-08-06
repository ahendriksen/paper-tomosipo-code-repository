#!/usr/bin/env python3

"""Benchmark the speed of FDK with various parameters

This demo requires the following packages:

```
conda install tomosipo=0.4.1 pytorch=1.8.1 cudatoolkit=X.X  tqdm -c astra-toolbox/label/dev -c pytorch -c defaults -c aahendriksen
```

"""

import torch
import tomosipo as ts
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import astra
from ts_algorithms import fdk

def time_function(f):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    f()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def benchmark(f, num_trials):
    timings = np.zeros(num_trials)

    for i in tqdm(range(num_trials)):
        timings[i] = time_function(f)

    return (
        timings.mean(),
        timings.std(),
        timings.min(),
        timings.max()
    )


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--num_gpus', default=1, type=int)

    parser.add_argument('--num_burnin', default=3, type=int)
    parser.add_argument('--num_trials', default=5, type=int)

    parser.add_argument('--astra', dest='use_astra_algorithm', action='store_true', default=False)
    parser.add_argument('--tomosipo', dest='use_astra_algorithm', action='store_false')
    parser.add_argument('--bp', default=False, action='store_true')

    # parse params
    args = parser.parse_args()

    print("Parameters")
    for k, v in args._get_kwargs():
        print(f"{k:<30} {v}")
    print()

    # Use gpus:
    astra.set_gpu_index(list(range(args.num_gpus)))

    # Define operator
    vg = ts.volume(shape=(1512, 1912, 1912))
    pg = ts.cone(
        angles=3600,
        shape=(1512, 1912),
        src_orig_dist=10_000,
        src_det_dist=10_000,
    )
    A = ts.operator(vg, pg)

    # Generate data:
    sino = torch.ones(A.range_shape)

    def only_bp():
        A.T(sino)

    def tomosipo_fdk():
        fdk(A, sino)

    def astra_fdk():
        vd = ts.data(vg)
        pd = ts.data(pg, sino)
        with vd, pd:
            ts.astra.fdk(vd, pd)

    if args.bp:
        f = only_bp
    else:
        f = astra_fdk if args.use_astra_algorithm else tomosipo_fdk

    print("Burning in.. ")
    _ = benchmark(f, args.num_burnin)

    print("Benchmark.. ")
    mean, std, min, max = benchmark(f, args.num_trials)

    print(f"Time (seconds): {mean:0.3f}+-{std:0.3f} in range ({min:0.3f} -- {max:0.3f})")
