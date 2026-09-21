from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal
from PIL import Image


gdal.UseExceptions()


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
PROCESSED_DIR = ROOT / "data" / "processed"
REGISTRATION_DIR = ROOT / "outputs" / "registration"
LOFTR_DIR = ROOT / "outputs" / "loftr"


# Existing scripts used by this pipeline.
PIPELINE_SCRIPTS = [
    "01_verify_dataset.py",
    "02_verify_pixels.py",
    "05_preprocess.py",
    "06_extract_overlap.py",
    "07_parse_tmc_oat.py",
    "10_tmc_overlap_geometry.py",
    "11_extract_tmc_overlap.py",
    "13_extract_ohrc_overlap.py",
    "16_common_grid.py",
    "17_ohrc_common_grid.py",
    "18_tmc_common_grid.py",
]


# Final paired images produced by scripts 17 and 18.
OHRC_COMMON_GRID = REGISTRATION_DIR / "OHRC_common_grid.tif"
TMC_COMMON_GRID = REGISTRATION_DIR / "TMC_common_grid.tif"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def run_script(script_name: str) -> None:
    """
    Execute one existing preprocessing script from the project root.
    """
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Missing preprocessing script: {script_path}")

    print(f"\n{'=' * 72}")
    print(f"Running {script_name}")
    print(f"{'=' * 72}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        check=True,
    )


def read_single_band(path: Path) -> np.ndarray:
    """
    Read a single-band GeoTIFF as float32.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected raster does not exist: {path}")

    ds = gdal.Open(str(path), gdal.GA_ReadOnly)

    if ds is None:
        raise RuntimeError(f"GDAL could not open: {path}")

    if ds.RasterCount < 1:
        raise RuntimeError(f"Raster has no bands: {path}")

    array = ds.GetRasterBand(1).ReadAsArray()

    if array is None:
        raise RuntimeError(f"Could not read raster data: {path}")

    return array.astype(np.float32)


def valid_mask(image: np.ndarray) -> np.ndarray:
    """
    Zero is treated as the invalid/padded-data value used in this repository.
    """
    return np.isfinite(image) & (image > 0)


def normalize_image(
    image: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robustly normalize a raster independently to [0, 1].

    Returns:
        normalized image: float32 array in [0, 1]
        valid mask: uint8 array containing 0 or 1
    """
    mask = valid_mask(image)

    if mask.sum() == 0:
        raise RuntimeError("Image contains no valid pixels.")

    values = image[mask]

    low = float(np.percentile(values, lower_percentile))
    high = float(np.percentile(values, upper_percentile))

    if not np.isfinite(low) or not np.isfinite(high):
        raise RuntimeError("Invalid percentile values found during normalization.")

    if high <= low:
        high = low + 1.0

    normalized = (image - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)

    # Keep invalid/padded pixels at zero.
    normalized[~mask] = 0.0

    return normalized, mask.astype(np.uint8)


def pad_to_multiple(
    image: np.ndarray,
    mask: np.ndarray,
    multiple: int = 8,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Pad image and mask on the bottom/right so dimensions are divisible by
    the requested multiple.

    LoFTR commonly uses coarse feature maps with downsampling. Padding to a
    multiple of 8 avoids incompatible spatial dimensions.
    """
    height, width = image.shape

    padded_height = int(np.ceil(height / multiple) * multiple)
    padded_width = int(np.ceil(width / multiple) * multiple)

    pad_bottom = padded_height - height
    pad_right = padded_width - width

    image_padded = np.pad(
        image,
        ((0, pad_bottom), (0, pad_right)),
        mode="constant",
        constant_values=0.0,
    )

    mask_padded = np.pad(
        mask,
        ((0, pad_bottom), (0, pad_right)),
        mode="constant",
        constant_values=0,
    )

    return image_padded, mask_padded, (pad_bottom, pad_right)


def save_png(image: np.ndarray, path: Path) -> None:
    """
    Save a normalized float image as an 8-bit grayscale PNG.
    """
    image_uint8 = np.round(image * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)


def validate_pair(
    ohrc: np.ndarray,
    tmc: np.ndarray,
    ohrc_mask: np.ndarray,
    tmc_mask: np.ndarray,
) -> None:
    """
    Validate that the two images are suitable as a paired input.
    """
    if ohrc.shape != tmc.shape:
        raise RuntimeError(
            "OHRC and TMC dimensions do not match: "
            f"OHRC={ohrc.shape}, TMC={tmc.shape}"
        )

    common_valid = (ohrc_mask > 0) & (tmc_mask > 0)
    common_pixels = int(common_valid.sum())

    if common_pixels == 0:
        raise RuntimeError("OHRC and TMC have no common valid pixels.")

    common_fraction = common_pixels / common_valid.size

    print("\nPAIR VALIDATION")
    print("-" * 72)
    print("Pair shape:", ohrc.shape)
    print("OHRC valid pixels:", int(ohrc_mask.sum()))
    print("TMC valid pixels :", int(tmc_mask.sum()))
    print("Common valid pixels:", common_pixels)
    print("Common valid fraction:", f"{common_fraction:.4f}")

    if common_fraction < 0.01:
        print(
            "WARNING: Less than 1% of the pair contains common valid pixels. "
            "Check the overlap geometry."
        )


def save_loftr_package(
    ohrc: np.ndarray,
    tmc: np.ndarray,
    ohrc_mask: np.ndarray,
    tmc_mask: np.ndarray,
    original_shape: tuple[int, int],
    padding: tuple[int, int],
) -> None:
    """
    Save files for later PyTorch/ResNet/LoFTR loading.
    """
    LOFTR_DIR.mkdir(parents=True, exist_ok=True)

    # NCHW format expected by most PyTorch image models.
    ohrc_tensor = ohrc[None, None, :, :].astype(np.float32)
    tmc_tensor = tmc[None, None, :, :].astype(np.float32)

    np.save(LOFTR_DIR / "ohrc.npy", ohrc_tensor)
    np.save(LOFTR_DIR / "tmc.npy", tmc_tensor)

    np.save(LOFTR_DIR / "ohrc_mask.npy", ohrc_mask[None, None, :, :])
    np.save(LOFTR_DIR / "tmc_mask.npy", tmc_mask[None, None, :, :])

    save_png(ohrc, LOFTR_DIR / "ohrc.png")
    save_png(tmc, LOFTR_DIR / "tmc.png")

    metadata = {
        "source_ohrc": str(OHRC_COMMON_GRID.relative_to(ROOT)),
        "source_tmc": str(TMC_COMMON_GRID.relative_to(ROOT)),
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

    print("\nSaved LoFTR package:")
    print("  ", LOFTR_DIR / "ohrc.npy")
    print("  ", LOFTR_DIR / "tmc.npy")
    print("  ", LOFTR_DIR / "ohrc_mask.npy")
    print("  ", LOFTR_DIR / "tmc_mask.npy")
    print("  ", LOFTR_DIR / "ohrc.png")
    print("  ", LOFTR_DIR / "tmc.png")
    print("  ", LOFTR_DIR / "metadata.json")


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main(run_existing_scripts: bool = True) -> None:
    if run_existing_scripts:
        for script_name in PIPELINE_SCRIPTS:
            run_script(script_name)

    print("\nReading common-grid rasters...")
    ohrc_raw = read_single_band(OHRC_COMMON_GRID)
    tmc_raw = read_single_band(TMC_COMMON_GRID)

    print("OHRC raw shape:", ohrc_raw.shape)
    print("TMC raw shape :", tmc_raw.shape)

    ohrc, ohrc_mask = normalize_image(ohrc_raw)
    tmc, tmc_mask = normalize_image(tmc_raw)

    validate_pair(ohrc, tmc, ohrc_mask, tmc_mask)

    original_shape = ohrc.shape

    # LoFTR coarse processing normally benefits from dimensions divisible by 8.
    ohrc, ohrc_mask, ohrc_padding = pad_to_multiple(
        ohrc,
        ohrc_mask,
        multiple=8,
    )

    tmc, tmc_mask, tmc_padding = pad_to_multiple(
        tmc,
        tmc_mask,
        multiple=8,
    )

    if ohrc.shape != tmc.shape:
        raise RuntimeError(
            "Padding produced different OHRC/TMC shapes: "
            f"OHRC={ohrc.shape}, TMC={tmc.shape}"
        )

    if ohrc_padding != tmc_padding:
        raise RuntimeError(
            "OHRC/TMC padding differs. The source pair is not geometrically aligned."
        )

    print("\nLoFTR input preparation")
    print("-" * 72)
    print("Original shape:", original_shape)
    print("Padded shape  :", ohrc.shape)
    print("Padding       :", ohrc_padding)

    save_loftr_package(
        ohrc=ohrc,
        tmc=tmc,
        ohrc_mask=ohrc_mask,
        tmc_mask=tmc_mask,
        original_shape=original_shape,
        padding=ohrc_padding,
    )

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the TMC/OHRC preprocessing pipeline and create paired "
            "LoFTR-ready inputs."
        )
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip scripts 01-18 and only convert existing "
            "OHRC_common_grid.tif and TMC_common_grid.tif files."
        ),
    )

    args = parser.parse_args()
    main(run_existing_scripts=not args.skip_existing)
