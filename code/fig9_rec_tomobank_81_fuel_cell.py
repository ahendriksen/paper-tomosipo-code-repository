#!/usr/bin/env python3

import numpy as np
import torch
import tifffile
import h5py
import tomosipo as ts
from ts_algorithms import fbp, sirt, tv_min2d
from tqdm import tqdm

f = h5py.File("/export/scratch3/hendriks/fuelcell_flat_fielded.h5", "r")

num_angles = 3600

zlim = slice(2, 3)
alim = slice(0, num_angles)

sino = torch.from_numpy(f["data"][alim, zlim, :]).transpose(0, 1).contiguous()
angles = torch.from_numpy(f["omega"][alim])

num_slices, num_angles, num_pixels = sino.shape
pg = ts.parallel(angles=1, shape=(num_slices, num_pixels))
vg = ts.volume(shape=(num_slices, num_pixels, num_pixels)).to_vec()

R = ts.rotate(pos=(0, 0, -17.3), axis=(1, 0, 0), angles=angles)

A = ts.operator(R * vg, pg)

print("fbp")
rec_fbp = fbp(A, sino)
print("sirt")
rec_sirt = sirt(A, sino.cuda(), num_iterations=200)

tifffile.imsave("tomobank_82_fbp.tif", rec_fbp.cpu().squeeze().numpy())
tifffile.imsave("tomobank_82_sirt.tif", rec_sirt.cpu().squeeze().numpy())

print("tv min")
rec_tv = tv_min2d(A, sino.cuda(), lam=5.62e-9 , num_iterations=500)

lams = 10 ** np.linspace(-9, -3, 10)
for lam in tqdm(lams):
    rec_tv = tv_min2d(A, sino.cuda(), lam=lam , num_iterations=500)
    tifffile.imsave(f"tomobank_82_tv_{lam:0.2e}.tif", rec_tv.cpu().squeeze().numpy())

# tifffile.imsave("tomobank_82_fbp.tif", rec_fbp.cpu().squeeze().numpy())
# tifffile.imsave("tomobank_82_sirt.tif", rec_sirt.cpu().squeeze().numpy())
# tifffile.imsave("tomobank_82_tv.tif", rec_tv.cpu().squeeze().numpy())
