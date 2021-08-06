#!/usr/bin/env python3

"""

A test script to give an indication of reconstruction speed of SIRT
applied to X-ray scattering tensor tomography.

Installation:

conda install cudatoolkit=10.2 pytorch=1.8.1 tomosipo=0.3.1 astra-toolbox tqdm  -c pytorch -c defaults -c astra-toolbox/label/dev -c aahendriksen

"""
from argparse import ArgumentParser
import numpy as np
from numpy.linalg import norm
import torch
import tomosipo as ts
import tomosipo.torch_support
from  tqdm import tqdm
from timeit import default_timer as timer

###############################################################################
#              Define projection geometry with tilt and rotation              #
###############################################################################
num_tilt = 46
num_rotation = 50
vol_shape = (44, 71, 71)
det_distance = 140
det_shape = (1100 // 9, 1440 // 9)

tilt_angles = np.linspace(0, np.pi / 4, num_tilt)
rotation_angles = np.linspace(0, 2 * np.pi, num_rotation, endpoint=False)

vg = ts.volume(shape=vol_shape).to_vec()
pg = ts.translate((0, det_distance, 0)) * ts.parallel(shape=det_shape).to_vec()

tilt = ts.rotate(pos=0, axis=(0, 0, 1), angles=tilt_angles)
rotate = ts.rotate(pos=0, axis=(1, 0, 0), angles=rotation_angles)
# For each tilt angle, perform a full rotation:
TR = ts.concatenate([tilt_single * rotate for tilt_single in tilt])

# *NOTE* volume is moving and detector remains static
A = ts.operator(TR * vg, pg)

###############################################################################
#                                 Determine ν                                 #
###############################################################################

# B is a single vector since we move the object and not the detector.
B = pg.ray_dir[0]

# The S vectors are defined in the frame of reference of the object
Ss = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
               [1, 1, 0], [1, 0, 1], [0, 1,1],
               [1, 1, 1]])
Ss = Ss / np.linalg.norm(Ss, axis=1, keepdims=True)

# The G vectors are defined in the "lab frame". They rotate along a half arc
# across the detector plane.

g_angles = np.linspace(-np.pi / 2, np.pi / 2, 8, endpoint=False)
Gs = ts.rotate(pos=0, axis=(0, 1, 0), angles=g_angles).transform_vec((1, 0, 0))

num_s = len(Ss)                 # 7
num_g = len(Gs)                 # 8

def calculate_nu(b, s, g, TR):
    nu = np.zeros(TR.num_steps)
    for j, s_rot in enumerate(TR.transform_vec(s)):
        nu[j] = (norm(np.cross(b, s_rot)) * np.dot(s_rot, g)) ** 2
    return nu

nu = torch.zeros(num_s, num_g, TR.num_steps)

for k in range(num_s):
    for i in range(num_g):
        nu_ki = calculate_nu(B, Ss[k], Gs[i], TR)
        nu[k, i] = torch.from_numpy(nu_ki).float()

###############################################################################
#                               fp, bp, and SIRT                              #
###############################################################################

def fp(x, nu):
    y = torch.zeros(num_g, *A.range_shape, device=x.device)
    for k in range(num_s):
        for i in range(num_g):
            y[i] += nu[k, i][None, :, None] * A(x[k])

    return y

def bp(y, nu):
    x = torch.zeros(num_s, *A.domain_shape, device=y.device)
    for k in range(num_s):
        for i in range(num_g):
            x[k] += A.T(nu[k, i][None, :, None] * y[i])
    return x

def sirt(y, nu, num_iterations=50):
    R = 1 / torch.clamp(fp(y.new_ones(num_s, *A.domain_shape), nu), min=ts.epsilon)
    C = 1 / torch.clamp(bp(y.new_ones(num_g, *A.range_shape), nu), min=ts.epsilon)

    x_rec = y.new_zeros(num_s, *A.domain_shape)

    for i in range(num_iterations):
        x_rec += C * bp(R * (y - fp(x_rec, nu)), nu)

    return x_rec

###############################################################################
#                             Benchmark functions                             #
###############################################################################
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



###############################################################################
#                          Reconstruction and timings                         #
###############################################################################

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--n_iters', default=50, type=int)
    parser.add_argument('--cpu', dest='gpu', action='store_false', default=True)
    parser.add_argument('--num_burnin', default=1, type=int)
    parser.add_argument('--num_trials', default=3, type=int)
    args = parser.parse_args()

    print("Parameters")
    for k, v in args._get_kwargs():
        print(f"{k:<30} {v}")
    print()


    dev = torch.device("cuda") if args.gpu else torch.device("cpu")
    y = torch.zeros(num_g, *A.range_shape, device=dev)

    def f():
        sirt(y.to(dev), nu.to(dev), args.n_iters)

    print("Burning in.. ")
    _ = benchmark(f, args.num_burnin)

    print("Benchmark.. ")
    mean, std, min, max = benchmark(f, args.num_trials)

    print(f"Time (seconds): {mean:0.3f}+-{std:0.3f} in range ({min:0.3f} -- {max:0.3f})")
