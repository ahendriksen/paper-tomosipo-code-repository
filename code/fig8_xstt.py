#!/usr/bin/env python3

"""

A test script to give an indication of reconstruction speed of SIRT
applied to X-ray scattering tensor tomography.

Installation:

conda install cudatoolkit=10.2 pytorch=1.8.1 tomosipo=0.3.1 astra-toolbox tqdm  -c pytorch -c defaults -c astra-toolbox/label/dev -c aahendriksen

"""
import numpy as np
from numpy.linalg import norm
import torch
import tomosipo as ts
from tomosipo.qt import animate
import tomosipo.torch_support
from  tqdm import tqdm
from timeit import default_timer as timer

###############################################################################
#              Define projection geometry with tilt and rotation              #
###############################################################################

visalize_opengl = False

num_tilt = 46
num_rotation = 50
vol_shape = (44, 71, 71)
det_distance = 240
det_shape = (1100 // 9, 1440 // 9)

tilt_angles = np.linspace(0, np.pi / 4, num_tilt, endpoint=True)
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

###############################################################################
#                                     Vis                                     #
###############################################################################
def lines(xs, pos=0):
    xs = np.array(xs, copy=False)
    return ts.volume_vec(shape=1, pos=pos, w=xs, u=0*xs, v=0*xs)


def points_to_curve(ps):
    ps = np.array(ps, copy=False)
    return ts.concatenate(
        [lines(np.array([q - p]), pos=np.array([(p + q) / 2])) for p, q in zip(ps, ps[1:])]
    )

S = ts.scale(1/75) * ts.rotate(pos=0, axis=(1, 0, 0), angles=[np.pi/10])

frame_num = 900

current_tilt = tilt[frame_num // num_rotation]
current_TR = current_tilt * rotate

rot_axis = lines([(200, 0, 0)])
tilt_axis = lines([(0, 0, 200)])

rot_axis_vol = lines([(vg.shape[0], 0, 0)])
tilt_axis_vol = lines([(0, 0, vg.shape[2])])

rot_curve = points_to_curve(current_TR.transform_point([(50, 25, 0)]))
tilt_curve = points_to_curve(tilt.transform_point([(25, 0, 50)]))

tilt_lines = tilt * lines([(50, 0, 0)], pos=(0, 0, 50))
current_tilt_line = current_tilt * lines([(100, 0, 0)], pos=(0, 0, 50))

vol_cell = vg[:, :10, :10, -10:]
unit_cell = pg[:, :20, -20:]

grating = ts.translate((0, det_distance / 2, 0)) * ts.parallel(shape=det_shape).to_vec()

beam = ts.volume(size=(80, det_distance * 2, 80)).to_vec()
beam2 = ts.volume(size=(80, det_distance, 80)).to_vec()

if visalize_opengl:
    animation = animate(
        S * TR[frame_num] * vg,
        S * TR[frame_num] * vol_cell,
        S * TR[frame_num] * rot_axis,
        S * tilt_axis,
        *(S * rot_curve),
        *(S * tilt_curve),
        *(S * tilt_lines),
        *(S * current_tilt_line),
        S * grating,
        S * pg,
        S * unit_cell,
        S * beam,
    )
    animation.window()

ts.svg(
    S * TR[frame_num] * vg,
    S * TR[frame_num] * vol_cell,
    S * TR[frame_num] * rot_axis,
    S * tilt_axis,
    S * TR[frame_num] * rot_axis_vol,
    S * tilt_axis_vol,
    *(S * rot_curve),
    *(S * tilt_curve),
    *(S * tilt_lines),
    *(S * current_tilt_line),
    S * grating,
    S * pg,
    S * unit_cell,
    S * beam,
    S * beam2,
).save("./imgs/fig8_xstt/raw.svg")

# S2 = ts.scale(1/150)
# TR_subsampled = ts.concatenate([tilt_single * rotate for tilt_single in tilt[::5]])

# if visalize_opengl:
#     animation = animate(
#         S2 * vg,
#         S2 * TR_subsampled.inv * pg,
#     )
#     animation.window()

# ts.svg(
#     S2 * vg,
#     S2 * TR_subsampled.inv * pg,
# ).save("./imgs/fig_rstt_perspective.svg")

###############################################################################
#                        Show unit cells and S vectors                        #
###############################################################################

S = ts.scale(0.75)

unit_vol = ts.volume(size=1).to_vec()
s_lines = lines(Ss)

Tv = ts.translate((0, 1, 0), alpha=np.linspace(-4, 4, 7))
ts.svg(
    *(S * Tv * s_lines),
    *(S * Tv * unit_vol),
).save("./imgs/fig8_xstt/unit_vols.svg")

g_lines = ts.rotate(pos=0, axis=(1, 0, 0), angles=np.pi/2) * lines(Gs)
unit_pg = ts.rotate(pos=0, axis=(1, 0, 0), angles=np.pi/2) * ts.parallel(size=1, shape=9).to_vec()

Tp = ts.translate((0, 1, 0), alpha=np.linspace(-4, 4, 8))
ts.svg(
    S * Tp[0] * unit_pg[:, :, 1],
    S * Tp[0] * unit_pg[:, :, 3],
    S * Tp[0] * unit_pg[:, :, 5],
    S * Tp[0] * unit_pg[:, :, 7],
    S * Tp[0] * unit_pg[:, 1, :],
    S * Tp[0] * unit_pg[:, 3, :],
    S * Tp[0] * unit_pg[:, 5, :],
    S * Tp[0] * unit_pg[:, 7, :],
    *(S * Tp * g_lines),
    *(S * Tp * unit_pg),
).save("./imgs/fig8_xstt/unit_cells.svg")
