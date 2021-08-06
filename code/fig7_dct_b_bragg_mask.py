import numpy as np
import tomosipo as ts
import matplotlib.pyplot as plt

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
det_size = (500, 500)
pg_static = ts.parallel(
    angles=np.zeros(num_angles),
    shape=det_shape,
    size=det_size
)
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

bragg_mask = np.empty((num_orientations, num_angles), dtype=bool)

for i in range(num_orientations):
    rotated_normal = R.transform_vec(plane_normals[i])
    for j in range(num_angles):
        bragg_mask[i, j] = in_bragg_condition(
            rotated_normal[j], incoming_ray_dir, bragg_angle
        )

fig = plt.figure(figsize=(4, 1.5))
plt.imshow(bragg_mask, cmap="binary")
plt.xlabel("Rotation angle index")
plt.ylabel("Orientation index")
plt.title("Bragg diffraction occurrence")
plt.gca().set_frame_on(False)
fig.tight_layout()
fig.savefig("./imgs/fig7_dct/bragg_mask.pdf")

print("Fraction of diffraction occurrence", bragg_mask.mean())
