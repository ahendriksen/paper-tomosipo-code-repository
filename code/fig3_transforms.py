#!/usr/bin/env python3

import tomosipo as ts
import numpy as np

# Translate
vg = ts.volume(size=1.0).to_vec()
T = ts.translate(
    axis=(0, 1, 0),
    alpha=[-1, 0.5, 2.0])
T_svg = ts.svg(T * vg)

T_svg.save("./imgs/fig3_transforms/translate.svg")

# Rotate
R = ts.rotate(
    pos=(0, 0, 0),
    axis=(1, 0, 0),
    angles=[0, np.pi / 3]
)
R_svg = ts.svg(R * vg)

R_svg.save("./imgs/fig3_transforms/rotate.svg")

# Scale
S = ts.scale(
    (1, 1, 1),
    alpha=[1, 1.5]
)
S_svg = ts.svg(S * vg)

S_svg.save("./imgs/fig3_transforms/scale.svg")

# Reflect
mirror = ts.volume(
    pos=(0, 1.2, 0),
    size=(2, 0, 2)
)
M = ts.reflect(
    pos=mirror.pos,
    axis=(0, 1, 0),
)
M_svg = ts.svg(vg, M * vg, mirror)
M_svg.save("./imgs/fig3_transforms/reflect.svg")

# Perspective
vg = ts.volume(size=0.5).to_vec()
pg = ts.cone(
    angles=100,
    size=2,
    src_orig_dist=3,
    src_det_dist=5
).to_vec()

P1_svg = ts.svg(vg, pg)
P1_svg.save("./imgs/fig3_transforms/P1.svg")

P = ts.from_perspective(vol=pg.to_vol())
T = ts.translate(pg.det_pos[0])

P2_svg = ts.svg(
    (T * P * vg)[::10],
    (T * P * pg)[::10],
)

P2_svg.save("./imgs/fig3_transforms/P2.svg")

ts.svg(P[::10] * vg).save("./imgs/fig3_transforms/P2_vol_multi.svg")
