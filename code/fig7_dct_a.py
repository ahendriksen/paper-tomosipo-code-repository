#!/usr/bin/env python3

import numpy as np
import tomosipo as ts


np.random.seed(1)

# Random sampling of the grain orientations on the unit sphere.
num_orientations = 20
plane_normals = np.random.normal(size=(num_orientations, 3))
plane_normals /= np.sqrt(np.sum(plane_normals ** 2, axis=1))[:, None]

# Rotation of the crystal
num_angles = 100
rot_angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

# Size and shape of static detector
det_shape = (100, 100)
det_size = (600, 800)
det_distance = 240

pg_static = ts.translate((0, det_distance, 0)) * ts.parallel(
    angles=np.zeros(num_angles),
    shape=det_shape,
    size=det_size
).to_vec()
incoming_ray_dir = pg_static.ray_dir[0]

# Rotation of the rotation stage
R = ts.rotate(pos=0, axis=(1, 0, 0), angles=rot_angles)

def diffracted_pg(pg_static, plane_normal, R):
    # 1. Rotate the plane normal
    rotated_plane_normal = R.transform_vec(plane_normal)
    # 2. Create a reflection in the rotating plane normal
    M = ts.reflect(pos=0, axis=rotated_plane_normal)
    # 3. Create a new vector geometry with dynamic ray direction
    return ts.parallel_vec(
        shape=pg_static.det_shape,
        ray_dir=M.transform_vec(pg_static.ray_dir),
        det_pos=pg_static.det_pos,
        det_v=pg_static.det_v,
        det_u=pg_static.det_u,
    )

# This block is *NOT* shown in the manuscript.

# The bragg angle must be in the interval (0, np.pi / 2)
bragg_angle = 0.1 * np.pi / 2

def in_bragg_condition(plane_normal, ray_dir, bragg_angle, epsilon=np.pi/80):
    norm_normal = np.sqrt(np.dot(plane_normal, plane_normal))
    norm_ray = np.sqrt(np.dot(ray_dir, ray_dir))

    snell_angle = np.arccos(
        np.dot(plane_normal, ray_dir) / (norm_normal * norm_ray)
    )
    return abs(snell_angle - (np.pi / 2 - bragg_angle)) < epsilon


vg = R * ts.volume(shape=100).to_vec()
diffracted_pgs = [
    diffracted_pg(pg_static, normal, R) for normal in plane_normals
]

###############################################################################
#                                Create Figure                                #
###############################################################################
def lines(xs, pos=0):
    xs = np.array(xs, copy=False)
    return ts.volume_vec(shape=1, pos=pos, w=xs, u=0*xs, v=0*xs)


def points_to_curve(ps):
    ps = np.array(ps, copy=False)
    return ts.concatenate(
        [lines(np.array([q - p]), pos=np.array([(p + q) / 2])) for p, q in zip(ps, ps[1:])]
    )

crystal_O = ts.rotate(pos=0, axis=(0.0, 1.0, 1.0), angles=0.35)
# Rotate the ground plane and z-axis to obtain the oriented crystal plane
# and normal vector
crystal_plane = crystal_O * ts.volume(pos=0,  size=(0.01, 100, 100)).to_vec()
crystal_normal = crystal_O * ts.volume(pos=(100, 0, 0), size=(200, 0., 0.01)).to_vec()

# Create a rotation operator around the Z axis
angles = np.linspace(0, 2 * np.pi, num_angles)
R = ts.rotate(pos=0, axis=(1, 0, 0), angles=angles)

# Create a rotation axis, and the rotated plane, and normal
# vector for display
rot_axis = lines([(400, 0, 0)])
rotated_plane = R * crystal_plane
rotated_normal = R * crystal_normal

# Create curve showing path of the plane normal (for visualization)
rotated_normal_curve = points_to_curve(rotated_normal.corners[:, 5])



frame_num = np.s_[80]


# 1. Rotate the plane normal
rotated_plane_normal = R.transform_vec(plane_normals[0])
# 2. Create a reflection in the rotating plane normal
M = ts.reflect(pos=0, axis=rotated_plane_normal)

in_beam = ts.volume(size=(100, 400, 100), pos=(0, -200, 0)).to_vec()
corners = in_beam.corners[0, [2, 3, 6, 7]]

d_pg = diffracted_pg(pg_static, crystal_normal.w, R)


projected_corner_coords = np.stack([d_pg.project_point(c) for c in corners])

def pg_coord_to_pos(pg, coords):
    return pg.det_pos + pg.det_v * coords[:, 0:1] + pg.det_u * coords[:, 1:]

projected_corners = np.stack([pg_coord_to_pos(d_pg, pcc) for pcc in projected_corner_coords])
rays = [ts.concatenate([points_to_curve([corners[i], pc]) for pc in projected_corners[i]]) for i in range(4)]


center_curve = points_to_curve(pg_coord_to_pos(d_pg, d_pg.project_point((0, 0, 0))))

crystal = ts.volume(shape=95, pos=(400, -500, -40))
grain = crystal[:40, :40, -40:].to_vec()

moved_grain = ts.translate((-200, 200, 20)) * grain

S = ts.scale(1/250) * ts.rotate(pos=0, axis=(1, 0, 0), angles=[np.pi/10])
ts.svg(
    S * in_beam,
    S * rot_axis,
    S * rotated_plane[frame_num],
    S * rotated_normal[frame_num],
    *(S * rotated_normal_curve),
    S * d_pg[frame_num],
    S * rays[0][frame_num],
    S * rays[1][frame_num],
    S * rays[2][frame_num],
    S * rays[3][frame_num],
    *(S * center_curve),
    S * grain,
    S * moved_grain,
    S * ts.translate(moved_grain.pos) * rotated_normal[frame_num],
    S * crystal.to_vec(),
).save("./imgs/fig7_dct/raw.svg")
