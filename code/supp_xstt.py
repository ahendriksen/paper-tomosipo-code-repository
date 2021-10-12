import numpy as np
from numpy.linalg import norm
import torch
import tomosipo as ts

###########################################################################
#                          Acquisition parameters                         #
###########################################################################
num_tilt = 46
num_rotation = 50
vol_shape = (44, 71, 71)
det_distance = 140
det_shape = (1100 // 9, 1440 // 9)

###########################################################################
#            Define projection geometry with tilt and rotation            #
###########################################################################
tilt_angles = -np.linspace(0, np.pi / 4, num_tilt)
rotation_angles = -np.linspace(0, 2 * np.pi, num_rotation, endpoint=False)

vg = ts.volume(shape=vol_shape).to_vec()
T_det = ts.translate((0, det_distance, 0))
pg = T_det * ts.parallel(shape=det_shape).to_vec()

tilt = ts.rotate(pos=0, axis=(0, 0, 1), angles=tilt_angles)
rotate = ts.rotate(pos=0, axis=(1, 0, 0), angles=rotation_angles)
# For each tilt angle, perform a full rotation:
TR = ts.concatenate([tilt_single * rotate for tilt_single in tilt])

# *NOTE* volume is moving and detector remains static
A = ts.operator(TR * vg, pg)

###########################################################################
#                           Compute B, S_k, G_i                           #
###########################################################################

# B is a single vector since we move the object and not the detector.
B = pg.ray_dir[0]

# The S vectors are defined in the frame of reference of the object
Ss = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1],
     [1, 1, 0], [1, 0, 1], [0, 1, 1],
     [1, 1, 1]]
)
Ss = Ss / np.linalg.norm(Ss, axis=1, keepdims=True)

# The G vectors are defined in the "lab frame". They rotate along a half
# arc across the detector plane.
g_angles = np.linspace(0, -np.pi, 8, endpoint=False)
R_G = ts.rotate(pos=0, axis=(0, 1, 0), angles=g_angles)
Gs = R_G.transform_vec((-1, 0, 0))

num_s = len(Ss)                 # 7
num_g = len(Gs)                 # 8


###########################################################################
#                               Determine nu                              #
###########################################################################
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


###########################################################################
#                             fp, bp, and SIRT                            #
###########################################################################
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
    R = 1 / torch.clamp(
        fp(y.new_ones(num_s, *A.domain_shape), nu),
        min=0.1,
    )
    C = 1 / torch.clamp(
        bp(y.new_ones(num_g, *A.range_shape), nu),
        min=ts.epsilon
    )

    x_rec = y.new_zeros(num_s, *A.domain_shape)

    for i in range(num_iterations):
        x_rec += C * bp(R * (y - fp(x_rec, nu)), nu)

    return x_rec

###########################################################################
#                              Reconstruction                             #
###########################################################################

x = torch.zeros(num_s, *A.domain_shape)

# FP on CPU:
y = fp(x, nu)

# FP on GPU:
y = fp(x.cuda(), nu.cuda())

# SIRT on CPU:
x_rec = sirt(y.cpu(), nu.cpu(), num_iterations=1)

# SIRT on GPU:
x_rec = sirt(y.cuda(), nu.cuda(), num_iterations=1)
