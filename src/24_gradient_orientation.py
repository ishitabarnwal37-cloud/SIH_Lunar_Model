from pathlib import Path
from osgeo import gdal
import numpy as np

gdal.UseExceptions()

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "outputs/registration"

def read(name):
    ds = gdal.Open(str(DIR / name))
    return ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

def orientation_features(a):
    mask = a > 0

    lo, hi = np.percentile(a[mask], (2, 98))
    a = np.clip((a - lo) / (hi - lo), 0, 1)

    gy, gx = np.gradient(a)

    mag = np.sqrt(gx * gx + gy * gy)
    angle = np.arctan2(gy, gx)

    # doubled angle makes opposite contrast edges equivalent
    c = np.cos(2 * angle) * mag
    s = np.sin(2 * angle) * mag

    c[~mask] = 0
    s[~mask] = 0

    return c, s

for name in ["OHRC", "TMC"]:
    a = read(f"{name}_common_grid.tif")
    c, s = orientation_features(a)

    np.save(DIR / f"{name}_ori_cos.npy", c)
    np.save(DIR / f"{name}_ori_sin.npy", s)

    print(name, c.shape)
    