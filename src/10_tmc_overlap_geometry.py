import numpy as np

# TMC system-level footprint: lat, lon
UL = (8.497995, 23.785752)
UR = (8.481320, 24.619795)
LL = (-25.815939, 22.876861)
LR = (-25.833864, 23.767155)

# OHRC 2024 right edge: the part inside TMC
OHRC = [
    (0.369695, 23.593739),
    (-0.444178, 23.575386)
]

def edge_lon(lat, top, bottom):
    t = (top[0] - lat) / (top[0] - bottom[0])
    return top[1] + t * (bottom[1] - top[1])

for lat, lon in OHRC:
    left = edge_lon(lat, UL, LL)
    right = edge_lon(lat, UR, LR)

    f = (lon - left) / (right - left)

    print(f"lat={lat:.6f}")
    print(f"TMC left edge  = {left:.6f}")
    print(f"OHRC right lon = {lon:.6f}")
    print(f"cross-track fraction = {f:.5f}")
    print()