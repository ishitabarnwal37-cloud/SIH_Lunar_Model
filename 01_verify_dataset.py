"""
Verify every Chandrayaan-2 product: label, dimensions, payload size.

Exit status depends only on the products required for the final OHRC-TMC
pair (OHRC_2024, TMC). Failures in OHRC_2021 / IIRS are reported as
non-blocking warnings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import (
    DATA_ROOT, PRODUCTS, print_status, validate_product,
)

print("Dataset location:", DATA_ROOT)
print("\nCHANDRAYAAN-2 DATASET VERIFICATION")
print("=" * 70)

blocking = []
for name, product in PRODUCTS.items():
    status = validate_product(name)
    print_status(status)
    if not status.ok:
        if product.required:
            blocking.append(name)
        else:
            print("Note    : optional product, does not block the OHRC-TMC pair")

print("\n" + "=" * 70)
if blocking:
    print("Verification FAILED for required product(s):", ", ".join(blocking))
    sys.exit(1)
print("Verification complete: all required products (OHRC_2024, TMC) are valid.")
