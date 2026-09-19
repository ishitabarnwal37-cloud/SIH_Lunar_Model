import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import REGISTRATION_DIR
from raster_io import read_array, write_raster

DIR = REGISTRATION_DIR

def read(name):
    return read_array(DIR / name).astype(np.float32)

a = read("OHRC_gradient_smooth.tif")
b = read("TMC_gradient_smooth.tif")

scores = []

for dy in range(-40, 41):
    for dx in range(-40, 41):

        ay1, ay2 = max(0, dy), min(a.shape[0], a.shape[0] + dy)
        ax1, ax2 = max(0, dx), min(a.shape[1], a.shape[1] + dx)

        by1, by2 = ay1 - dy, ay2 - dy
        bx1, bx2 = ax1 - dx, ax2 - dx

        A = a[ay1:ay2, ax1:ax2]
        B = b[by1:by2, bx1:bx2]

        mask = (A > 0) & (B > 0)

        if mask.sum() < 1000:
            continue

        A = A[mask]
        B = B[mask]

        A -= A.mean()
        B -= B.mean()

        denom = np.sqrt(np.sum(A * A) * np.sum(B * B))

        if denom == 0:
            continue

        score = np.sum(A * B) / denom
        scores.append((float(score), dx, dy))

scores.sort(reverse=True)

for rank, (score, dx, dy) in enumerate(scores[:10], 1):
    print(f"{rank:2d}: dx={dx:3d}  dy={dy:3d}  score={score:.5f}")