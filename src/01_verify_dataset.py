from pathlib import Path
import subprocess

# ============================================================
# ORIGINAL CHANDRAYAAN-2 DATASET LOCATION
# ============================================================

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folder inside the project
DATASET = PROJECT_ROOT / "Dataset"

print("Dataset location:", DATASET)

# ============================================================
# PDS4 PRODUCT LABELS
# ============================================================

products = {

    "IIRS": DATASET /
        "ch2_iir_nri_20240523T1600301891_d_img_d18" /
        "data/raw/20240523" /
        "ch2_iir_nri_20240523T1600301891_d_img_d18.xml",

    "OHRC_2021": DATASET /
        "ch2_ohr_ncp_20210402T0546284043_d_img_d18" /
        "data/calibrated/20210402" /
        "ch2_ohr_ncp_20210402T0546284043_d_img_d18.xml",

    "OHRC_2024": DATASET /
        "ch2_ohr_ncp_20240330T0035085365_d_img_d18" /
        "data/calibrated/20240330" /
        "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml",

    "TMC": DATASET /
        "ch2_tmc_ndn_20240523T1600309548_d_oth_d18" /
        "data/derived/20240523" /
        "ch2_tmc_ndn_20240523T1600309548_d_oth_d18.xml"
}

# ============================================================
# VERIFY PRODUCTS
# ============================================================

print("\nCHANDRAYAAN-2 DATASET VERIFICATION")
print("=" * 70)

for name, path in products.items():

    print(f"\n{name}")
    print("-" * 70)

    print("Path:", path)
    print("Exists:", path.exists())

    if not path.exists():
        print("ERROR: File not found")
        continue

    result = subprocess.run(
        ["gdalinfo", str(path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("GDAL ERROR:")
        print(result.stderr)
        continue

    for line in result.stdout.splitlines():

        if (
            line.startswith("Driver:")
            or line.startswith("Size is")
            or line.startswith("Band ")
        ):
            print(line)

print("\n" + "=" * 70)
print("Verification complete.")