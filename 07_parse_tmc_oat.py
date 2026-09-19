import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import DATA_ROOT

OAT = DATA_ROOT / "ch2_tmc_ndn_20240523T1600309548_d_oth_d18/miscellaneous/derived/20240523/ch2_tmc_ndn_20240523T1600309548_d_oth_d18.oat"

if not OAT.is_file():
    sys.exit(f"ERROR: TMC orbit/attitude table not found: {OAT}")

targets = {
    "OHRC_TOP": 0.371,
    "OHRC_CENTER": -0.036,
    "OHRC_BOTTOM": -0.443,
}

records = []

with open(OAT) as f:
    for line in f:
        p = line.split()

        record = int(p[1])
        lat = float(p[30])       # payload latitude
        lon = float(p[31])       # payload longitude

        records.append((record, lat, lon))

for name, target_lat in targets.items():
    best = min(records, key=lambda r: abs(r[1] - target_lat))
    print(
        f"{name:12s}  "
        f"target_lat={target_lat:9.6f}  "
        f"record={best[0]:5d}  "
        f"lat={best[1]:10.6f}  "
        f"lon={best[2]:10.6f}"
    )