from pathlib import Path
from osgeo import gdal
import numpy as np

gdal.UseExceptions()

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "outputs/registration"

files = {
    "OHRC": DIR / "OHRC_common_grid.tif",
    "TMC": DIR / "TMC_common_grid.tif"
}

def gradient(src):
    ds = gdal.Open(str(src))
    a = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    mask = a > 0

    lo, hi = np.percentile(a[mask], (2, 98))
    a = np.clip((a - lo) / (hi - lo), 0, 1)

    gy, gx = np.gradient(a)
    g = np.sqrt(gx * gx + gy * gy)

    g[~mask] = 0

    if np.any(g > 0):
        p = np.percentile(g[g > 0], 99)
        g = np.clip(g / p, 0, 1)

    return (g * 255).astype(np.uint8)

for name, src in files.items():
    img = gradient(src)

    out_path = DIR / f"{name}_gradient.tif"

    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        str(out_path),
        img.shape[1],
        img.shape[0],
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW"]
    )

    out.GetRasterBand(1).WriteArray(img)
    out.FlushCache()
    out = None

    print(name, img.shape, "->", out_path)