from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "outputs/overlap"
OUT_DIR = ROOT / "outputs/registration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ohrc = cv2.imread(str(IN_DIR / "OHRC_2024_overlap_candidate.png"), 0)
tmc = cv2.imread(str(IN_DIR / "TMC_OHRC_narrow_candidate.png"), 0)

if ohrc is None or tmc is None:
    raise RuntimeError("Could not read candidate images")

target_h = 3000

def resize_height(img, h):
    scale = h / img.shape[0]
    w = round(img.shape[1] * scale)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

ohrc = resize_height(ohrc, target_h)
tmc = resize_height(tmc, target_h)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

ohrc = clahe.apply(ohrc)
tmc = clahe.apply(tmc)

cv2.imwrite(str(OUT_DIR / "OHRC_reg.png"), ohrc)
cv2.imwrite(str(OUT_DIR / "TMC_reg.png"), tmc)

print("OHRC:", ohrc.shape)
print("TMC :", tmc.shape)
print("Saved to:", OUT_DIR)
