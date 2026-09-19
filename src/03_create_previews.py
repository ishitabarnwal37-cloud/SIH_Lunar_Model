from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "Dataset"
OUTPUT = PROJECT_ROOT / "outputs" / "previews"

OUTPUT.mkdir(parents=True, exist_ok=True)

products = {

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

print("\nCREATING PREVIEWS")
print("=" * 60)

for name, source in products.items():

    destination = OUTPUT / f"{name}_preview.tif"

    print(f"\nCreating {name} preview...")

    command = [
        "gdal_translate",

        # Output format
        "-of", "GTiff",

        # Read only band 1
        "-b", "1",

        # Reduce width to 1000 px.
        # Height is calculated automatically.
        "-outsize", "1000", "0",

        # Automatic display scaling
        "-scale",

        # Compression
        "-co", "COMPRESS=LZW",

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
        print("Saved:", destination)
    else:
        print("FAILED")
        print(result.stderr)

print("\n" + "=" * 60)
print("Preview generation complete.")