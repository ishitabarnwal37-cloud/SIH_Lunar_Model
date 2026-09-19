"""
Full science-pixel read test: every byte of every payload is read.

Exit status depends only on the required OHRC_2024 and TMC products.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import PRODUCTS, validate_product

print("\nFULL SCIENCE-PIXEL READ TEST")
print("=" * 60)

blocking = []
for name, product in PRODUCTS.items():
    print(f"\nTesting {name}...")
    status = validate_product(name, deep=True)
    if status.ok:
        lab = status.label
        print(f"PASS - all pixels readable ({lab.samples:,} x {lab.lines:,}, "
              f"{status.actual_bytes:,} bytes, {status.storage})")
        continue

    print(f"FAIL - {status.message}")
    for line in status.describe():
        print("       " + line)
    if product.required:
        blocking.append(name)
    else:
        print("       (optional product, does not block the OHRC-TMC pair)")

print("\n" + "=" * 60)
if blocking:
    print("Read test FAILED for required product(s):", ", ".join(blocking))
    sys.exit(1)
print("Read test complete: all required products are readable.")
