import tomosipo as ts
import numpy as np
t = np.linspace(-1, 1, 100)     # Time
s = 2 * np.pi * t               # Angle
radius = 2                      # Radius of helix
h = 1.0                         # Vertical "speed"

R = ts.rotate(pos=0, axis=(1, 0, 0), angles=s)
T = ts.translate(axis=(1, 0, 0), alpha = h * s / (2 * np.pi))
H = T * R

vg = ts.volume()
pg = ts.cone(size=2, src_orig_dist=radius, src_det_dist=2 * radius).to_vec()
ts.svg(vg, H * pg).save("./imgs/fig5_helical/raw.svg")
