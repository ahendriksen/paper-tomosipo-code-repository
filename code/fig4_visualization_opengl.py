#!/usr/bin/env python3

import tomosipo as ts
import numpy as np
from tomosipo.qt import animate

N = 100
# your particle
vg = ts.volume().to_vec()
# A detector (usually a parallel beam detector, but a cone-beam detector looks
# nicer in the animations)
flip180 = ts.rotate(pos=0, axis=(1, 0, 0), deg=180)
pg = flip180 * ts.cone(size=20, src_orig_dist=10, src_det_dist=20).to_vec()

# Rotate into a uniformly random orientation:
Rs = [
    ts.rotate(pos=0, axis=np.random.normal(size=3), rad=np.random.uniform(0, 2 * np.pi))
    for _ in range(N)
]

# Move to a random location:
Ts = [
    ts.translate(np.random.uniform(-5, 5, size=3) * np.array([1, 0, 1]))
    for _ in range(N)
]

# Scale geometries to fit in video:
visual_S = ts.scale(1 / 3)
animation = animate(
    *[visual_S * T * R * vg for (T, R) in zip(Ts, Rs)],
    visual_S * pg
)

# Start interative visualization
animation.window()
