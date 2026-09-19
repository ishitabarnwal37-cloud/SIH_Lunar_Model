"""
Create working GeoTIFF rasters in data/processed/ from the PDS4 products.

    python src/05_preprocess.py                       # all products that validate
    python src/05_preprocess.py --products OHRC_2024 TMC

Required products (OHRC_2024, TMC) must be valid: an invalid one stops the
script with a non-zero exit code. Invalid optional products (OHRC_2021, IIRS)
are skipped with a message.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import PROCESSED_DIR, PRODUCTS, validate_product
from raster_io import copy_band

# (output name, product, band). Band 137 is the selected IIRS working band.
WORKING_RASTERS = [
    ("OHRC_2021", "OHRC_2021", 1),
    ("OHRC_2024", "OHRC_2024", 1),
    ("TMC",       "TMC",       1),
    ("IIRS_B137", "IIRS",      137),
]

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--products", nargs="+", choices=list(PRODUCTS),
                    help="products to convert (default: every product that validates)")
args = parser.parse_args()

selected = [r for r in WORKING_RASTERS if args.products is None or r[1] in args.products]

print("\nPREPROCESSING WORKING RASTERS")
print("=" * 60)

failed_required = []

for out_name, product_name, band in selected:
    product = PRODUCTS[product_name]
    print(f"\nProcessing {out_name} (band {band})")

    status = validate_product(product_name)
    if not status.ok:
        if product.required or args.products is not None:
            print(f"FAILED - {status.message}")
            failed_required.append(out_name)
        else:
            print(f"SKIPPED (optional product is invalid) - {status.message}")
        continue

    destination = PROCESSED_DIR / f"{out_name}.tif"
    payload = status.label.payload
    # Raw PDS4 payloads are read through their label; GeoTIFFs directly.
    source = payload if payload.suffix.lower() in {".tif", ".tiff"} else status.label.path
    try:
        # Keep original numeric values; no scaling here.
        height, width = copy_band(source, destination, band=band)
    except Exception as error:  # noqa: BLE001 - report the real cause and stop
        print(f"FAILED - {type(error).__name__}: {error}")
        failed_required.append(out_name)
        continue

    print(f"SUCCESS  {width:,} x {height:,} -> {destination}")

print("\n" + "=" * 60)
if failed_required:
    print("FAILED to create:", ", ".join(failed_required))
    sys.exit(1)
print("Working rasters created.")
