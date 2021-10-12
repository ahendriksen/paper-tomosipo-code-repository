import numpy as np
import tomosipo as ts

np.random.seed(1)

###########################################################################
#                          Acquisition parameters                         #
###########################################################################

# The bragg angle must be in the interval (0, np.pi / 2)
bragg_angle = 0.1 * np.pi / 2

num_orientations = 20
num_angles = 100                # number of rotation angles over 2 pi arc

# Size and shape of static detector
det_shape = (100, 100)
det_size = (600, 800)
det_distance = 240

###########################################################################
#                             Static geometry                             #
###########################################################################

# Random sampling of the grain orientations on the unit sphere.
plane_normals = np.random.normal(size=(num_orientations, 3))
plane_normals /= np.sqrt(np.sum(plane_normals ** 2, axis=1))[:, None]


pg_static = ts.translate((0, det_distance, 0)) * ts.parallel(
    angles=np.zeros(num_angles),
    shape=det_shape,
    size=det_size
).to_vec()
incoming_ray_dir = pg_static.ray_dir[0]

###########################################################################
#                     Diffraction and Bragg condition                     #
###########################################################################

def diffracted_pg(pg_static, plane_normal, R):
    # 1. Rotate the plane normal
    rotated_plane_normal = R.transform_vec(plane_normal)
    # 2. Create a reflection in the rotating plane normal
    M = ts.reflect(pos=0, axis=rotated_plane_normal)
    # 3. Create a new vector geometry with dynamic ray direction
    return ts.parallel_vec(
        shape=pg_static.det_shape,
        ray_dir=M.transform_vec(pg_static.ray_dir), # Reflect ray direction
        det_pos=pg_static.det_pos,
        det_v=pg_static.det_v,
        det_u=pg_static.det_u,
    )


def in_bragg_condition(plane_normal, ray_dir, bragg_angle, epsilon=np.pi/80):
    norm_normal = np.sqrt(np.dot(plane_normal, plane_normal))
    norm_ray = np.sqrt(np.dot(ray_dir, ray_dir))

    snell_angle = np.arccos(
        np.dot(plane_normal, ray_dir) / (norm_normal * norm_ray)
    )
    return abs(snell_angle - (np.pi / 2 - bragg_angle)) < epsilon


###########################################################################
#                            Rotating geometry                            #
###########################################################################
# Rotation of the rotation stage
rot_angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
R = ts.rotate(pos=0, axis=(1, 0, 0), angles=rot_angles)

vg = R * ts.volume(shape=100).to_vec()
# Create a list of geometries using a list comprehension
diffracted_pgs = [
    diffracted_pg(pg_static, normal, R) for normal in plane_normals
]

###########################################################################
#                                Bragg mask                               #
###########################################################################
bragg_mask = np.empty((num_orientations, num_angles), dtype=bool)

for i in range(num_orientations):
    # 1. Rotate the plane normal associated with the lattice orientation
    rotated_normal = R.transform_vec(plane_normals[i])
    # 2. Determine whether the rotated plane normal is in Bragg condition
    for j in range(num_angles):
        bragg_mask[i, j] = in_bragg_condition(
            rotated_normal[j], incoming_ray_dir, bragg_angle
        )

###########################################################################
#               Operators and projection and backprojection               #
###########################################################################

# Compute an operator per orientation
operators = [
    ts.operator(vg[bragg_mask[i]], diffracted_pgs[i][bragg_mask[i]])
    for i in range(num_orientations)
]

def fp(x):
    y = np.zeros((det_shape[0], num_angles, det_shape[1]), dtype=np.float32)
    for i in range(num_orientations):
        mask = bragg_mask[i]
        A = operators[i]
        y[:, mask] += A(x[i])
    return y

def bp(y):
    x = np.zeros((num_orientations, *vg.shape), dtype=np.float32)
    for x_oriented, A, mask in zip(x, operators, bragg_mask):
        x_oriented[:] += A.T(y[:, mask])
    return x

x = np.zeros((num_orientations, *vg.shape), dtype=np.float32)
y = fp(x)
bp = bp(y)
