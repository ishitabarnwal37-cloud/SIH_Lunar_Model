from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "Dataset"

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

print("\nFULL SCIENCE-PIXEL READ TEST")
print("=" * 60)

for name, path in products.items():

    print(f"\nTesting {name}...")

    result = subprocess.run(
        ["gdalinfo", "-checksum", str(path)],
        capture_output=True,
        text=True
    )

    errors = [
        line for line in result.stderr.splitlines()
        if "ERROR" in line.upper()
    ]

    if result.returncode == 0 and not errors:
        print("PASS - science pixels readable")
    else:
        print("FAIL")

        if errors:
            for error in errors:
                print(error)
        else:
            print(result.stderr)

print("\n" + "=" * 60)
print("Read test complete.")
