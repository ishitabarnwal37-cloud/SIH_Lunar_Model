from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "Dataset"
OUTPUT = PROJECT_ROOT / "data" / "processed"

OUTPUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# INPUT PRODUCTS
# ------------------------------------------------------------

OHRC_2021 = (
    DATASET /
    "ch2_ohr_ncp_20210402T0546284043_d_img_d18" /
    "data/calibrated/20210402" /
    "ch2_ohr_ncp_20210402T0546284043_d_img_d18.xml"
)

OHRC_2024 = (
    DATASET /
    "ch2_ohr_ncp_20240330T0035085365_d_img_d18" /
    "data/calibrated/20240330" /
    "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml"
)

TMC = (
    DATASET /
    "ch2_tmc_ndn_20240523T1600309548_d_oth_d18" /
    "data/derived/20240523" /
    "ch2_tmc_ndn_20240523T1600309548_d_oth_d18.xml"
)

IIRS = (
    DATASET /
    "ch2_iir_nri_20240523T1600301891_d_img_d18" /
    "data/raw/20240523" /
    "ch2_iir_nri_20240523T1600301891_d_img_d18.xml"
)

# ------------------------------------------------------------
# WORKING DATA
# ------------------------------------------------------------

products = [
    ("OHRC_2021", OHRC_2021, 1),
    ("OHRC_2024", OHRC_2024, 1),
    ("TMC",       TMC,       1),

    # Selected IIRS working band
    ("IIRS_B137", IIRS, 137),
]

print("\nPREPROCESSING WORKING RASTERS")
print("=" * 60)

for name, source, band in products:

    destination = OUTPUT / f"{name}.tif"

    print(f"\nProcessing {name}")
    print(f"Band: {band}")

    command = [
        "gdal_translate",

        "-of", "GTiff",
        "-b", str(band),

        # Keep original numeric values
        # NO -scale here

        "-co", "TILED=YES",
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=YES",

        str(source),
        str(destination)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("SUCCESS")
        print(destination)
    else:
        print("FAILED")
        print(result.stderr)

print("\n" + "=" * 60)
print("Working rasters created.")
