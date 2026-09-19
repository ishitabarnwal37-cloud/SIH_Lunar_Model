from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "Dataset"
OUTPUT = PROJECT_ROOT / "outputs" / "previews"

IIRS = (
    DATASET /
    "ch2_iir_nri_20240523T1600301891_d_img_d18" /
    "data/raw/20240523" /
    "ch2_iir_nri_20240523T1600301891_d_img_d18.xml"
)

# Exact/near-exact representative bands from XML wavelength sequence
# Band 1 = 712.3 nm
# spacing ≈ 16.85 nm

bands = {
    "1000nm": 18,     # ~998.8 nm
    "1500nm": 48,     # ~1504.4 nm
    "2000nm": 78,     # ~2010.0 nm
    "3000nm": 137,    # ~3004.3 nm
}

# Examine a central 1000-row section
XOFF = 0
YOFF = 6000
WIDTH = 250
HEIGHT = 1000

for name, band in bands.items():

    destination = OUTPUT / f"IIRS_{name}_closeup.tif"

    print(f"Creating {name} - band {band}")

    result = subprocess.run(
        [
            "gdal_translate",
            "-of", "GTiff",

            "-b", str(band),

            # Native-pixel crop
            "-srcwin",
            str(XOFF),
            str(YOFF),
            str(WIDTH),
            str(HEIGHT),

            # Contrast stretch
            "-scale",

            "-co", "COMPRESS=LZW",

            str(IIRS),
            str(destination)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("SUCCESS:", destination)
    else:
        print("FAILED")
        print(result.stderr)