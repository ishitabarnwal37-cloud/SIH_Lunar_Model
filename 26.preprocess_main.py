"""
Final OHRC 2024 + TMC preprocessing pipeline -> paired LoFTR-ready tensors.

Only the two required products are used (OHRC_2024, TMC); OHRC_2021 and IIRS
are never needed and never block this pipeline.

    python src/26.preprocess_main.py               # full run
    python src/26.preprocess_main.py --skip-existing
        # only convert the existing OHRC/TMC common-grid rasters

Exit status is 0 only if every stage ran AND the exported package passed the
read-back verification at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import (  # noqa: E402
    COMMON_GRID_HEIGHT, COMMON_GRID_WIDTH, FINAL_PRODUCTS, LOFTR_DIR,
    PRODUCTS, REGISTRATION_DIR, WORK_ROOT, require_valid,
)
from raster_io import read_array  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent

# (script, extra arguments). Product validation happens in-process first.
PIPELINE_SCRIPTS = [
    ("05_preprocess.py", ["--products", *FINAL_PRODUCTS]),   # OHRC_2024 + TMC -> GeoTIFF
    ("06_extract_overlap.py", []),   # TMC overlap window (IIRS part is optional)
    ("07_parse_tmc_oat.py", []),     # TMC OAT geometry report
    ("10_tmc_overlap_geometry.py", []),
    ("11_extract_tmc_overlap.py", []),   # TMC corridor
    ("13_extract_ohrc_overlap.py", []),  # OHRC corridor
    ("16_common_grid.py", []),
    ("17_ohrc_common_grid.py", []),      # OHRC -> common grid
    ("18_tmc_common_grid.py", []),       # TMC  -> common grid
]

# Final paired images produced by scripts 17 and 18.
OHRC_COMMON_GRID = REGISTRATION_DIR / "OHRC_common_grid.tif"
TMC_COMMON_GRID = REGISTRATION_DIR / "TMC_common_grid.tif"

LOFTR_FILES = [
    "ohrc.npy", "tmc.npy", "ohrc_mask.npy", "tmc_mask.npy",
    "ohrc.png", "tmc.png", "metadata.json",
]


class PipelineError(RuntimeError):
    """A stage failed; the message says which and why."""


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(WORK_ROOT))
    except ValueError:
        return str(path)


def run_script(script_name: str, args: list[str]) -> None:
    """Run one preprocessing script; any non-zero exit aborts the pipeline."""
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        raise PipelineError(f"Missing preprocessing script: {script_path}")

    print(f"\n{'=' * 72}\nRunning {script_name} {' '.join(args)}\n{'=' * 72}", flush=True)

    result = subprocess.run(
        [sys.executable, str(script_path), *args], cwd=str(WORK_ROOT),
    )
    if result.returncode != 0:
        raise PipelineError(f"{script_name} failed with exit code {result.returncode}")


def remove_stale_package() -> None:
    """Delete any LoFTR files from an earlier run so they can't pass as new."""
    for name in LOFTR_FILES:
        (LOFTR_DIR / name).unlink(missing_ok=True)


def read_common_grid(path: Path) -> np.ndarray:
    if not path.exists():
        raise PipelineError(f"Expected raster does not exist: {path}")
    array = read_array(path)
    if array.shape != (COMMON_GRID_HEIGHT, COMMON_GRID_WIDTH):
        raise PipelineError(
            f"{path.name} is {array.shape[1]} x {array.shape[0]} (w x h); the common "
            f"grid must be {COMMON_GRID_WIDTH} x {COMMON_GRID_HEIGHT}."
        )
    return array.astype(np.float32)


def valid_mask(image: np.ndarray) -> np.ndarray:
    """Zero is the invalid/padded-data value used in this repository."""
    return np.isfinite(image) & (image > 0)


def normalize_image(
    image: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly normalize a raster to [0, 1]; returns (image, uint8 valid mask)."""
    mask = valid_mask(image)
    if mask.sum() == 0:
        raise PipelineError("Image contains no valid pixels.")

    values = image[mask]
    low = float(np.percentile(values, lower_percentile))
    high = float(np.percentile(values, upper_percentile))
    if not np.isfinite(low) or not np.isfinite(high):
        raise PipelineError("Invalid percentile values found during normalization.")
    if high <= low:
        high = low + 1.0

    normalized = np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)
    normalized[~mask] = 0.0   # keep invalid/padded pixels at zero
    return normalized, mask.astype(np.uint8)


def pad_to_multiple(
    image: np.ndarray, mask: np.ndarray, multiple: int = 8,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Pad bottom/right so both dimensions are divisible by `multiple` (LoFTR stride)."""
    height, width = image.shape
    pad_bottom = -height % multiple
    pad_right = -width % multiple
    pads = ((0, pad_bottom), (0, pad_right))
    return (
        np.pad(image, pads, mode="constant", constant_values=0.0),
        np.pad(mask, pads, mode="constant", constant_values=0),
        (pad_bottom, pad_right),
    )


def save_png(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.round(image * 255.0).clip(0, 255).astype(np.uint8), mode="L").save(path)


def validate_pair(ohrc_mask: np.ndarray, tmc_mask: np.ndarray) -> float:
    """Check the pair overlaps; returns the common valid fraction."""
    if ohrc_mask.shape != tmc_mask.shape:
        raise PipelineError(
            f"OHRC and TMC dimensions do not match: {ohrc_mask.shape} vs {tmc_mask.shape}"
        )
    common = (ohrc_mask > 0) & (tmc_mask > 0)
    fraction = float(common.mean())

    print("\nPAIR VALIDATION")
    print("-" * 72)
    print("Pair shape         :", ohrc_mask.shape)
    print("OHRC valid pixels  :", int(ohrc_mask.sum()))
    print("TMC valid pixels   :", int(tmc_mask.sum()))
    print("Common valid pixels:", int(common.sum()))
    print("Common valid frac. :", f"{fraction:.4f}")

    if common.sum() == 0:
        raise PipelineError("OHRC and TMC have no common valid pixels.")
    if fraction < 0.01:
        print("WARNING: less than 1% of the pair is common valid data; check the overlap geometry.")
    return fraction


def save_loftr_package(ohrc, tmc, ohrc_mask, tmc_mask, original_shape, padding) -> None:
    """Write NCHW tensors, masks, PNG previews and metadata for LoFTR."""
    LOFTR_DIR.mkdir(parents=True, exist_ok=True)

    np.save(LOFTR_DIR / "ohrc.npy", ohrc[None, None].astype(np.float32))
    np.save(LOFTR_DIR / "tmc.npy", tmc[None, None].astype(np.float32))
    np.save(LOFTR_DIR / "ohrc_mask.npy", ohrc_mask[None, None])
    np.save(LOFTR_DIR / "tmc_mask.npy", tmc_mask[None, None])
    save_png(ohrc, LOFTR_DIR / "ohrc.png")
    save_png(tmc, LOFTR_DIR / "tmc.png")

    metadata = {
        "source_ohrc": rel(OHRC_COMMON_GRID),
        "source_tmc": rel(TMC_COMMON_GRID),
        "source_products": {name: PRODUCTS[name].folder for name in FINAL_PRODUCTS},
        "original_shape": list(original_shape),
        "padded_shape": list(ohrc.shape),
        "padding_bottom_right": list(padding),
        "tensor_format": "NCHW",
        "dtype": "float32",
        "value_range": "[0, 1]",
        "image0": "OHRC",
        "image1": "TMC",
        "invalid_value": 0.0,
        "recommended_usage": (
            "Use these as paired spatial images. Do not apply global average "
            "pooling before LoFTR."
        ),
    }
    with open(LOFTR_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def verify_loftr_package() -> None:
    """Read every exported file back and check shapes/dtypes; raise on any problem."""
    problems = []

    for name in LOFTR_FILES:
        if not (LOFTR_DIR / name).is_file():
            problems.append(f"missing output: {name}")
    if problems:
        raise PipelineError("; ".join(problems))

    arrays = {n: np.load(LOFTR_DIR / f"{n}.npy") for n in ("ohrc", "tmc", "ohrc_mask", "tmc_mask")}
    meta = json.loads((LOFTR_DIR / "metadata.json").read_text(encoding="utf-8"))

    for name in ("ohrc", "tmc"):
        a = arrays[name]
        if a.ndim != 4 or a.shape[:2] != (1, 1):
            problems.append(f"{name}.npy is not NCHW (1,1,H,W): {a.shape}")
        if a.dtype != np.float32:
            problems.append(f"{name}.npy dtype is {a.dtype}, expected float32")
        if a.min() < 0.0 or a.max() > 1.0:
            problems.append(f"{name}.npy values outside [0,1]")
        if not np.isfinite(a).all():
            problems.append(f"{name}.npy contains NaN/inf")
    for name in ("ohrc_mask", "tmc_mask"):
        a = arrays[name]
        if a.shape != arrays["ohrc"].shape:
            problems.append(f"{name}.npy shape {a.shape} != image shape {arrays['ohrc'].shape}")
        if not np.isin(a, (0, 1)).all():
            problems.append(f"{name}.npy is not binary")
    if arrays["tmc"].shape != arrays["ohrc"].shape:
        problems.append("ohrc.npy and tmc.npy shapes differ")

    _, _, height, width = arrays["ohrc"].shape
    if height % 8 or width % 8:
        problems.append(f"padded size {width} x {height} is not divisible by 8")
    if meta.get("padded_shape") != [height, width]:
        problems.append("metadata.json padded_shape does not match the tensors")
    for name in ("ohrc", "tmc"):
        with Image.open(LOFTR_DIR / f"{name}.png") as im:
            if im.size != (width, height):
                problems.append(f"{name}.png size {im.size} != tensor size {(width, height)}")

    if problems:
        raise PipelineError("Exported package failed verification: " + "; ".join(problems))

    print("\nVERIFIED OUTPUT PACKAGE")
    print("-" * 72)
    for name in LOFTR_FILES:
        print(f"  {rel(LOFTR_DIR / name)}")
    print("Tensor shape (NCHW):", arrays["ohrc"].shape, "| mask shape:", arrays["ohrc_mask"].shape)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main(run_existing_scripts: bool = True) -> None:
    remove_stale_package()

    if run_existing_scripts:
        print("Validating required products:", ", ".join(FINAL_PRODUCTS))
        for status in require_valid(FINAL_PRODUCTS):   # exits(2) with details if invalid
            print(f"  {status.name}: PASS ({status.actual_bytes:,} bytes, {status.storage})")

        for script_name, args in PIPELINE_SCRIPTS:
            run_script(script_name, args)

    print("\nReading common-grid rasters...")
    ohrc_raw = read_common_grid(OHRC_COMMON_GRID)
    tmc_raw = read_common_grid(TMC_COMMON_GRID)
    print("OHRC raw shape:", ohrc_raw.shape)
    print("TMC raw shape :", tmc_raw.shape)

    ohrc, ohrc_mask = normalize_image(ohrc_raw)
    tmc, tmc_mask = normalize_image(tmc_raw)
    validate_pair(ohrc_mask, tmc_mask)

    original_shape = ohrc.shape
    ohrc, ohrc_mask, ohrc_padding = pad_to_multiple(ohrc, ohrc_mask, multiple=8)
    tmc, tmc_mask, tmc_padding = pad_to_multiple(tmc, tmc_mask, multiple=8)

    if ohrc.shape != tmc.shape or ohrc_padding != tmc_padding:
        raise PipelineError("Padding produced different OHRC/TMC shapes; pair is not aligned.")

    print("\nLoFTR input preparation")
    print("-" * 72)
    print("Original shape:", original_shape)
    print("Padded shape  :", ohrc.shape)
    print("Padding       :", ohrc_padding)

    save_loftr_package(ohrc, tmc, ohrc_mask, tmc_mask, original_shape, ohrc_padding)
    verify_loftr_package()

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the OHRC 2024 + TMC preprocessing pipeline and create paired LoFTR-ready inputs."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip product validation and scripts 05-18; only convert the existing "
             "OHRC_common_grid.tif and TMC_common_grid.tif.",
    )
    options = parser.parse_args()

    try:
        main(run_existing_scripts=not options.skip_existing)
    except PipelineError as error:
        print(f"\nPIPELINE FAILED: {error}", file=sys.stderr)
        sys.exit(1)
