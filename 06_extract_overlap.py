import sys
from pathlib import Path

from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import OVERLAP_DIR, PROCESSED_DIR
from raster_io import copy_band, open_raster

OUTPUT = OVERLAP_DIR
DATA = PROCESSED_DIR

# ============================================================
# OHRC 2024 FOOTPRINT
# This is our reference geographic region.
# ============================================================

OHRC_CORNERS = {
    "UL": (0.372032, 23.472939),
    "UR": (0.369695, 23.593739),
    "LL": (-0.441842, 23.454679),
    "LR": (-0.444178, 23.575386),
}

# ============================================================
# TMC SYSTEM-LEVEL CORNERS
# ============================================================

TMC_CORNERS = {
    "UL": (8.497995, 23.785752),
    "UR": (8.481320, 24.619795),
    "LL": (-25.815939, 22.876861),
    "LR": (-25.833864, 23.767155),
}

# ============================================================
# IIRS CORNERS
# ============================================================

IIRS_CORNERS = {
    "UL": (8.195483, 23.771330),
    "UR": (8.179012, 24.604061),
    "LL": (-26.147187, 22.857987),
    "LR": (-26.165032, 23.750095),
}

# ============================================================
# Bilinear geographic model
# ============================================================

def interpolate(corners, x, y):

    ul_lat, ul_lon = corners["UL"]
    ur_lat, ur_lon = corners["UR"]
    ll_lat, ll_lon = corners["LL"]
    lr_lat, lr_lon = corners["LR"]

    lat = (
        (1-x)*(1-y)*ul_lat +
        x*(1-y)*ur_lat +
        (1-x)*y*ll_lat +
        x*y*lr_lat
    )

    lon = (
        (1-x)*(1-y)*ul_lon +
        x*(1-y)*ur_lon +
        (1-x)*y*ll_lon +
        x*y*lr_lon
    )

    return lat, lon


def geographic_to_pixel(lat_target, lon_target, corners,
                        width, height):

    # Numerical search for approximate inverse mapping.
    # Coarse-to-fine search keeps memory usage negligible.

    best_x = 0.5
    best_y = 0.5
    search_radius = 0.5

    for _ in range(8):

        best_error = float("inf")
        new_best_x = best_x
        new_best_y = best_y

        steps = 30

        for iy in range(steps + 1):

            y = best_y - search_radius + \
                2 * search_radius * iy / steps

            y = max(0.0, min(1.0, y))

            for ix in range(steps + 1):

                x = best_x - search_radius + \
                    2 * search_radius * ix / steps

                x = max(0.0, min(1.0, x))

                lat, lon = interpolate(corners, x, y)

                error = (
                    (lat - lat_target) ** 2 +
                    (lon - lon_target) ** 2
                )

                if error < best_error:
                    best_error = error
                    new_best_x = x
                    new_best_y = y

        best_x = new_best_x
        best_y = new_best_y
        search_radius /= 5

    pixel_x = int(round(best_x * (width - 1)))
    pixel_y = int(round(best_y * (height - 1)))

    return pixel_x, pixel_y


# ============================================================
# Determine bounding window containing all four OHRC corners
# ============================================================

def calculate_window(corners, width, height):

    pixels = []

    for name, (lat, lon) in OHRC_CORNERS.items():

        x, y = geographic_to_pixel(
            lat,
            lon,
            corners,
            width,
            height
        )

        pixels.append((x, y))

        print(
            f"OHRC {name}: "
            f"lat={lat:.6f}, lon={lon:.6f} "
            f"-> pixel ({x}, {y})"
        )

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]

    xoff = min(xs)
    yoff = min(ys)

    xend = max(xs)
    yend = max(ys)

    # Add a safety margin
    margin_x = max(10, int((xend - xoff) * 0.10))
    margin_y = max(10, int((yend - yoff) * 0.10))

    xoff = max(0, xoff - margin_x)
    yoff = max(0, yoff - margin_y)

    xend = min(width - 1, xend + margin_x)
    yend = min(height - 1, yend + margin_y)

    win_width = xend - xoff + 1
    win_height = yend - yoff + 1

    return xoff, yoff, win_width, win_height


# ============================================================
# Extract window
# ============================================================

def extract(name, source, corners, required):
    """
    Cut the OHRC-overlap window out of `source`. Raster size is read from the
    file itself. Returns True on success. A missing/unreadable optional
    product is skipped; a required one is a failure.
    """
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    if not source.exists():
        print(f"{'FAILED' if required else 'SKIPPED'} - missing working raster: {source}")
        return not required

    with open_raster(source) as ds:
        width, height = ds.width, ds.height
    print(f"Raster size: {width} x {height}")

    xoff, yoff, win_width, win_height = calculate_window(corners, width, height)

    print()
    print("Extraction window:")
    print("x offset :", xoff)
    print("y offset :", yoff)
    print("width    :", win_width)
    print("height   :", win_height)

    destination = OUTPUT / f"{name}_OHRC2024_overlap.tif"
    try:
        copy_band(source, destination, window=Window(xoff, yoff, win_width, win_height))
    except Exception as error:  # noqa: BLE001
        print(f"FAILED - {type(error).__name__}: {error}")
        return False

    print("SUCCESS")
    print(destination)
    return True


# ============================================================
# RUN
# ============================================================

results = {
    # TMC is required; IIRS is optional and skipped unless IIRS_B137.tif exists.
    "TMC": extract("TMC", DATA / "TMC.tif", TMC_CORNERS, required=True),
    "IIRS_B137": extract("IIRS_B137", DATA / "IIRS_B137.tif", IIRS_CORNERS, required=False),
}

if not results["TMC"] or not results["IIRS_B137"]:
    print("\nFAILED:", ", ".join(k for k, v in results.items() if not v))
    sys.exit(1)

print("\nDone.")
