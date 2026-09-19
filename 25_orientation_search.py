import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import REGISTRATION_DIR

DIR = REGISTRATION_DIR

ac = np.load(DIR / "OHRC_ori_cos.npy")
as_ = np.load(DIR / "OHRC_ori_sin.npy")
bc = np.load(DIR / "TMC_ori_cos.npy")
bs = np.load(DIR / "TMC_ori_sin.npy")

scores = []

for dy in range(-40, 41):
    for dx in range(-40, 41):

        ay1 = max(0, dy)
        ay2 = min(ac.shape[0], ac.shape[0] + dy)
        ax1 = max(0, dx)
        ax2 = min(ac.shape[1], ac.shape[1] + dx)

        by1, by2 = ay1 - dy, ay2 - dy
        bx1, bx2 = ax1 - dx, ax2 - dx

        AC = ac[ay1:ay2, ax1:ax2]
        AS = as_[ay1:ay2, ax1:ax2]
        BC = bc[by1:by2, bx1:bx2]
        BS = bs[by1:by2, bx1:bx2]

        ma = np.sqrt(AC * AC + AS * AS)
        mb = np.sqrt(BC * BC + BS * BS)

        mask = (ma > 0) & (mb > 0)

        if mask.sum() < 1000:
            continue

        dot = AC[mask] * BC[mask] + AS[mask] * BS[mask]
        denom = ma[mask] * mb[mask]

        valid = denom > 0

        if valid.sum() < 1000:
            continue

        score = np.mean(dot[valid] / denom[valid])

        scores.append((float(score), dx, dy))

scores.sort(reverse=True)

for rank, (score, dx, dy) in enumerate(scores[:10], 1):
    print(f"{rank:2d}: dx={dx:3d}  dy={dy:3d}  score={score:.5f}")