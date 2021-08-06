#!/usr/bin/env python3

"""Benchmark the speed of SIRT with various parameters

This demo requires the following packages:

```
conda install tomosipo=0.4.1 pytorch=1.8.1 cudatoolkit=X.X  tqdm -c astra-toolbox/label/dev -c pytorch -c defaults -c aahendriksen
```

To see the difference between ASTRA's SIRT3D_CUDA and a tomosipo
implementation of SIRT, run:

> python benchmark_sirt.py --N 128 --n_iter=200 --astra
> python benchmark_sirt.py --N 128 --n_iter=200 --tomosipo

To see the effect of storing intermediate data in RAM, rather than in
GPU memory, run:

> python benchmark_sirt.py --N 128 --n_iter=200 --tomosipo --pingpong # on CPU
> python benchmark_sirt.py --N 128 --n_iter=200 --tomosipo --no-pingpong # on GPU

Of course, if data is too big, you might run out of memory, as this
example demonstrates:

> python benchmark_sirt.py --N 1024 --n_iter=200 --tomosipo --pingpong
> python benchmark_sirt.py --N 1024 --n_iter=200 --tomosipo --no-pingpong


Obtained on a dual-socket system with a Titan RTX 2080 Ti GPU,
preliminary results suggest that the tomosipo implementation is
slightly faster than the ASTRA implementation:

> python benchmark_sirt.py --N 256 --num_burnin 1 --num_trials 5 --n_iter=200 --tomosipo
: Time (seconds): 17.914+-0.076 in range (17.780 -- 17.984)

> python benchmark_sirt.py --N 256 --num_burnin 1 --num_trials 5 --n_iter=200 --astra
: Time (seconds): 19.288+-0.051 in range (19.195 -- 19.339)

We should note that at very small data sizes, the overhead of Python
does influence performance and ASTRA is faster again:

> python benchmark_sirt.py --N 32 --num_burnin 5 --num_trials 10 --n_iter=200 --tomosipo
: Time (seconds): 0.397+-0.005 in range (0.393 -- 0.409)

> python benchmark_sirt.py --N 32 --num_burnin 5 --num_trials 10 --n_iter=200 --astra
: Time (seconds): 0.244+-0.001 in range (0.243 -- 0.246)

"""

import torch
import tomosipo as ts
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import astra


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


def parallel_operator(N = 512):
    N = 512
    vg = ts.volume(size=1, shape=N)
    pg = ts.parallel(angles=3 * N // 2, shape=(N, 3 * N // 2), size=(1, 1.5))
    return ts.operator(vg, pg)


def helical_operator(N=512):
    t = np.linspace(-1, 1, N)   # Time
    s = 2 * np.pi * t           # Angle
    radius = 2                  # Radius of helix
    h = 1.0                     # Vertical "speed"

    R = ts.rotate(pos=0, axis=(1, 0, 0), angles=s)
    T = ts.translate(axis=(1, 0, 0), alpha=h * s / (2 * np.pi))
    H = T * R

    vg = ts.volume(shape=N)
    pg = ts.cone(
        shape=3 * N // 2,
        size=2,
        src_orig_dist=radius,
        src_det_dist=2 * radius
    )
    return ts.operator(vg, H * pg.to_vec())


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--N', default=512, type=int)
    parser.add_argument('--n_iters', default=10, type=int)
    parser.add_argument('--pingpong', dest='pingpong', action='store_true', default=False)
    parser.add_argument('--no-pingpong', dest='pingpong', action='store_false')

    parser.add_argument('--num_burnin', default=3, type=int)
    parser.add_argument('--num_trials', default=5, type=int)

    parser.add_argument('--astra', dest='use_astra_algorithm', action='store_true', default=False)
    parser.add_argument('--tomosipo', dest='use_astra_algorithm', action='store_false')

    parser.add_argument('--geometry', default="parallel", type=str)

    # parse params
    args = parser.parse_args()

    print("Parameters")
    for k, v in args._get_kwargs():
        print(f"{k:<30} {v}")
    print()

    # Generate data:
    if args.geometry == "parallel":
        A = parallel_operator(args.N)
    elif args.geometry == "helical":
        A = helical_operator(args.N)
    else:
        raise ValueError(f"Unrecognized geometry: {args.geometry}. Expected 'parallel' or 'helical'")

    sino = torch.zeros(A.range_shape)

    dev = torch.device("cpu") if args.pingpong else torch.device("cuda")

    def tomosipo_sirt():
        C = A.T(torch.ones(A.range_shape, device=dev))
        C[C < ts.epsilon] = np.Inf
        C.reciprocal_()

        R = A(torch.ones(A.domain_shape, device=dev))
        R[R < ts.epsilon] = np.Inf
        R.reciprocal_()

        y = sino.to(dev)
        x_cur = torch.zeros(A.domain_shape, device=dev)

        for _ in range(args.n_iters):
            x_cur -= C * A.T(R * (A(x_cur) - y))

        # Move result to cpu to keep comparison with ASTRA fair.
        x_cur = x_cur.cpu()

    def astra_sirt():
        vd = ts.data(A.volume_geometry)
        pd = ts.data(A.projection_geometry, sino)
        with vd, pd:
            rec_id = vd.to_astra()
            sinogram_id = pd.to_astra()
            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sinogram_id

            # Create the algorithm object from the configuration structure
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, args.n_iters)
            astra.algorithm.delete(alg_id)

    f = astra_sirt if args.use_astra_algorithm else tomosipo_sirt

    print("Burning in.. ")
    _ = benchmark(f, args.num_burnin)

    print("Benchmark.. ")
    mean, std, min, max = benchmark(f, args.num_trials)

    print(f"Time (seconds): {mean:0.3f}+-{std:0.3f} in range ({min:0.3f} -- {max:0.3f})")
