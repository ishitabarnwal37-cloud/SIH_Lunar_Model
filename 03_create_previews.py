"""
Create 1000-px-wide 8-bit previews of the OHRC / TMC products.
Optional diagnostic: invalid products are skipped, not treated as errors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import PREVIEW_DIR, PRODUCTS, validate_product
from raster_io import write_preview

print("\nCREATING PREVIEWS")
print("=" * 60)

errors = []

for name in ("OHRC_2021", "OHRC_2024", "TMC"):
    print(f"\nCreating {name} preview...")

    status = validate_product(name)
    if not status.ok:
        print(f"SKIPPED - {status.message}")
        if PRODUCTS[name].required:
            errors.append(name)
        continue

    payload = status.label.payload
    source = payload if payload.suffix.lower() in {".tif", ".tiff"} else status.label.path
    destination = PREVIEW_DIR / f"{name}_preview.tif"
    try:
        write_preview(source, destination, band=1, out_width=1000)
    except Exception as error:  # noqa: BLE001
        print(f"FAILED - {type(error).__name__}: {error}")
        errors.append(name)
        continue
    print("SUCCESS")
    print("Saved:", destination)

print("\n" + "=" * 60)
if errors:
    print("Preview generation FAILED for:", ", ".join(errors))
    sys.exit(1)
print("Preview generation complete.")
