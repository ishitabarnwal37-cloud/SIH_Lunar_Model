"""
IIRS close-up previews at representative wavelengths.

Optional: IIRS is not part of the final OHRC-TMC pair. If the IIRS cube is
truncated this script reports why and exits 0 without writing anything.
"""

import sys
from pathlib import Path

from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import PREVIEW_DIR, validate_product
from raster_io import write_preview

status = validate_product("IIRS")
if not status.ok:
    print(f"SKIPPED IIRS previews - {status.message}")
    for line in status.describe():
        print("  " + line)
    print("IIRS is optional and does not block the OHRC-TMC pair.")
    sys.exit(0)

# Exact/near-exact representative bands from XML wavelength sequence
# Band 1 = 712.3 nm, spacing ~ 16.85 nm
bands = {
    "1000nm": 18,     # ~998.8 nm
    "1500nm": 48,     # ~1504.4 nm
    "2000nm": 78,     # ~2010.0 nm
    "3000nm": 137,    # ~3004.3 nm
}

# Examine a central 1000-row section (native pixels)
window = Window(0, 6000, 250, 1000)
errors = []

for name, band in bands.items():
    destination = PREVIEW_DIR / f"IIRS_{name}_closeup.tif"
    print(f"Creating {name} - band {band}")
    try:
        write_preview(status.label.path, destination, band=band, window=window)
    except Exception as error:  # noqa: BLE001
        print(f"FAILED - {type(error).__name__}: {error}")
        errors.append(name)
        continue
    print("SUCCESS:", destination)

if errors:
    sys.exit(1)
