#!/usr/bin/env python3

import tomosipo as ts
import numpy as np

def lines(xs, pos=0):
    xs = np.array(xs, copy=False)
    return ts.volume_vec(shape=1, pos=pos, w=xs, u=0*xs, v=0*xs)


# Volume geometry
vg = ts.volume(
    shape=(2, 2, 2),
    size=(2, 2, 2),
    pos=(0, 0, 0),
)

vg_svg = ts.svg(vg, vg[1:, :1, 1:])
vg_svg.save("./imgs/fig2_geometries/volume.svg")

# Single-axis parallel beam
angles = np.linspace(
    0, np.pi, 5,
    endpoint=False
)
par_pg = ts.parallel(
    angles=angles,
    shape=(2, 2),
    size=(2, 2),
)
par_svg = ts.svg(par_pg)
par_svg.save("./imgs/fig2_geometries/parallel.svg")

# Circular cone beam
cone_pg = ts.cone(
    angles=100,
    shape=2,
    src_orig_dist=1,
    src_det_dist=4,
)
cone_svg = ts.svg(cone_pg)
cone_svg.save("./imgs/fig2_geometries/cone.svg")

# Volume vector geometry
vg_vec = ts.volume_vec(
    shape=(2, 2, 2),
    pos=[(0, 0, 0)],
    w=[(1, 0, 0)],
    v=[(0, 1, 0)],
    u=[(0, 0, 1)],
)
vg_vec_svg = ts.svg(vg_vec, vg_vec[:, 1:, :1, 1:])
vg_vec_svg.save("./imgs/fig2_geometries/volume_vec.svg")

# Parallel vector geometry
par_pg_vec = ts.parallel_vec(
    shape=(2, 2),
    ray_dir=[(0, 1,  0)],
    det_pos=[(0, 2, 0)],
    det_v=[(1, 0, 0)],
    det_u=[(0, 0, 1)],
)
par_pg_vec_svg = ts.svg(
    par_pg_vec,
    par_pg_vec[:, :1, :1],
    ts.volume(size=(2, 4, 2), pos=(0, 0, 0)),
    lines(par_pg_vec.ray_dir[0], pos=par_pg_vec.det_pos),
)
par_pg_vec_svg.save("./imgs/fig2_geometries/parallel_vec.svg")

# Cone vector geometry
cone_pg_vec = ts.cone_vec(
    shape=(2, 2),
    src_pos=[(0, -2, 0)],
    det_pos=[(0, 2, 0)],
    det_v=[(1, 0, 0)],
    det_u=[(0, 0, 1)],
)
cone_pg_vec_svg = ts.svg(
    cone_pg_vec,
    cone_pg_vec[:, :1, :1],
)
cone_pg_vec_svg.save("./imgs/fig2_geometries/cone_vec.svg")
