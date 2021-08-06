import tomosipo as ts
import numpy as np

# Detector properties
pixel_size = 0.02
det_shape = np.array((120, 180))
det_size = pixel_size * det_shape
det_pos = (0, 2, 0)

# Volume properties
vol_shape = np.array((80, 80, 80))
vol_size = pixel_size * vol_shape

# Rotation properties
num_angles = 50
angles = np.linspace(0, np.pi, num_angles, endpoint=False)
rot_axis_pos = (0, 0, 0.3)
z_axis = (1, 0, 0)

def lines(xs, pos=0):
    xs = np.array(xs, copy=False)
    return ts.volume_vec(shape=1, pos=pos, w=xs, u=0*xs, v=0*xs)


def points_to_curve(ps):
    ps = np.array(ps, copy=False)
    return ts.concatenate(
        [lines(np.array([q - p]), pos=np.array([(p + q) / 2])) for p, q in zip(ps, ps[1:])]
    )

# Static Detector at custom position
T = ts.translate(det_pos)
pg = ts.parallel(size=det_size, shape=det_shape)
pg = T * pg.to_vec()

# Rotating volume
R = ts.rotate(pos=rot_axis_pos, axis=z_axis, angles=angles)
vg_static = ts.volume(pos=0, shape=vol_shape, size=vol_size)
vg = R * vg_static.to_vec()

# Create operator
A = ts.operator(vg, pg)

# Create figure
rot_axis = lines([(3, 0, 0)], pos=rot_axis_pos)
rot_axis_on_vol = lines([(vg_static.size[0], 0, 0)], pos=rot_axis_pos)
rot_axis_on_det = T * lines([(pg.det_size[0], 0, 0)], pos=rot_axis_pos)
vol_center = lines([(vg_static.size[0], 0, 0)], pos=0)
center_of_det = T * lines([(pg.det_size[0], 0, 0)], pos=0)
center_curve = points_to_curve((R * vol_center).corners[:, 5])
# corner_curve = points_to_curve(vg.corners[:, 6])

V = ts.translate((0, -1, 0))
svg = ts.svg(
    V * vg[0],
    # ts.translate((0.1, 0, 0)) * center_curve[0],
    # vg[-10],
    # ts.translate((0.1, 0, 0)) * center_curve[-10],
    V * pg,
    V * rot_axis,
    V * rot_axis_on_vol,
    V * rot_axis_on_det,
    V * center_of_det,
    # vol_center,
    *(V * center_curve),
    # *corner_curve,
)
svg.save("./imgs/fig6_parallel_beam_lab_frame/raw.svg")
