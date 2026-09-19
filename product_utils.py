"""
Shared paths, PDS4 label parsing and payload validation for the
OHRC 2024 + TMC preprocessing pipeline.

Environment overrides (used for isolated testing; default = project root):
    LUNAR_DATA_ROOT  folder that contains the ch2_* product folders
    LUNAR_WORK_ROOT  folder under which data/processed and outputs/ are written
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import xml.etree.ElementTree as ET
except ImportError as error:  # e.g. MSYS2 Python with a broken pyexpat DLL
    sys.exit(
        f"ERROR: this Python cannot parse XML ({error}).\n"
        "Use the project environment instead:  .venv\\Scripts\\python.exe <script>"
    )


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("LUNAR_DATA_ROOT", PROJECT_ROOT)).resolve()
WORK_ROOT = Path(os.environ.get("LUNAR_WORK_ROOT", PROJECT_ROOT)).resolve()

PROCESSED_DIR = WORK_ROOT / "data" / "processed"
OUTPUTS_DIR = WORK_ROOT / "outputs"
PREVIEW_DIR = OUTPUTS_DIR / "previews"
OVERLAP_DIR = OUTPUTS_DIR / "overlap"
REGISTRATION_DIR = OUTPUTS_DIR / "registration"
LOFTR_DIR = OUTPUTS_DIR / "loftr"

# Intended common registration grid (rows, columns). Scripts 11, 13, 17
# and 18 all produce or consume rasters of exactly this size.
COMMON_GRID_HEIGHT = 4340
COMMON_GRID_WIDTH = 141


# ---------------------------------------------------------------------
# Product registry
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Product:
    name: str
    folder: str
    level: str      # calibrated / derived / raw
    date: str
    required: bool  # required for the final OHRC-TMC pair

    @property
    def label(self) -> Path:
        return (
            DATA_ROOT / self.folder / "data" / self.level / self.date
            / f"{self.folder}.xml"
        )


PRODUCTS = {
    p.name: p
    for p in (
        Product("OHRC_2024", "ch2_ohr_ncp_20240330T0035085365_d_img_d18",
                "calibrated", "20240330", True),
        Product("TMC", "ch2_tmc_ndn_20240523T1600309548_d_oth_d18",
                "derived", "20240523", True),
        Product("OHRC_2021", "ch2_ohr_ncp_20210402T0546284043_d_img_d18",
                "calibrated", "20210402", False),
        Product("IIRS", "ch2_iir_nri_20240523T1600301891_d_img_d18",
                "raw", "20240523", False),
    )
}

FINAL_PRODUCTS = tuple(name for name, p in PRODUCTS.items() if p.required)


# ---------------------------------------------------------------------
# PDS4 label parsing
# ---------------------------------------------------------------------

PDS4_BYTES_PER_ELEMENT = {
    "UnsignedByte": 1, "SignedByte": 1,
    "UnsignedLSB2": 2, "UnsignedMSB2": 2, "SignedLSB2": 2, "SignedMSB2": 2,
    "UnsignedLSB4": 4, "UnsignedMSB4": 4, "SignedLSB4": 4, "SignedMSB4": 4,
    "IEEE754LSBSingle": 4, "IEEE754MSBSingle": 4,
    "IEEE754LSBDouble": 8, "IEEE754MSBDouble": 8,
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str:
    for child in element.iter():
        if local_name(child.tag) == name and child.text:
            return child.text.strip()
    return ""


@dataclass(frozen=True)
class Label:
    path: Path
    payload: Path
    lines: int
    samples: int
    bands: int
    data_type: str
    offset_bytes: int
    declared_bytes: int   # <file_size> from the label (0 if absent)

    @property
    def bytes_per_element(self) -> int:
        return PDS4_BYTES_PER_ELEMENT[self.data_type]

    @property
    def expected_bytes(self) -> int:
        """Bytes of pixel data alone: offset + lines * samples * bands * itemsize."""
        return (
            self.offset_bytes
            + self.lines * self.samples * self.bands * self.bytes_per_element
        )


def parse_label(label: Path) -> Label:
    """Parse a namespaced PDS4 label; raises FileNotFoundError / ValueError."""
    if not label.is_file():
        raise FileNotFoundError(f"Missing product label: {label}")

    try:
        root = ET.parse(label).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Label is not valid XML ({label}): {error}") from error

    file_name = declared = None
    for element in root.iter():
        if local_name(element.tag) == "File":
            file_name = _child_text(element, "file_name")
            declared = _child_text(element, "file_size")
            break
    if not file_name:
        raise ValueError(f"No payload file_name in label: {label}")

    array = next(
        (e for e in root.iter()
         if local_name(e.tag) in {"Array_2D_Image", "Array_3D_Spectrum", "Array_3D_Image"}),
        None,
    )
    if array is None:
        raise ValueError(f"No Array_2D_Image / Array_3D_* element in label: {label}")

    axes = {}
    for axis in array:
        if local_name(axis.tag) != "Axis_Array":
            continue
        name = _child_text(axis, "axis_name").lower()
        elements = _child_text(axis, "elements")
        if name and elements:
            axes[name] = int(elements)
    if "line" not in axes or "sample" not in axes:
        raise ValueError(f"Missing Line/Sample dimensions in label: {label}")

    data_type = _child_text(array, "data_type")
    if data_type not in PDS4_BYTES_PER_ELEMENT:
        raise ValueError(
            f"Unsupported PDS4 data_type {data_type!r} in {label}; "
            "add it to PDS4_BYTES_PER_ELEMENT."
        )

    return Label(
        path=label,
        payload=label.parent / file_name,
        lines=axes["line"],
        samples=axes["sample"],
        bands=axes.get("band", 1),
        data_type=data_type,
        offset_bytes=int(_child_text(array, "offset") or 0),
        declared_bytes=int(declared) if declared else 0,
    )


# ---------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------

@dataclass
class ProductStatus:
    name: str
    ok: bool
    message: str
    label: Label | None = None
    actual_bytes: int = 0
    storage: str = ""   # "raw" | "tiff-uncompressed" | "tiff-compressed"

    def describe(self) -> list[str]:
        """Human-readable detail lines (exact expected/actual sizes)."""
        lines = []
        lab = self.label
        if lab is not None:
            lines.append(f"Payload : {lab.payload}")
            lines.append(
                f"Label   : {lab.samples:,} samples x {lab.lines:,} lines"
                f" x {lab.bands} band(s), {lab.data_type}"
            )
            lines.append(f"Expected: {lab.expected_bytes:,} bytes (pixel data)")
            if lab.declared_bytes:
                lines.append(f"Declared: {lab.declared_bytes:,} bytes (label file_size)")
            lines.append(f"Actual  : {self.actual_bytes:,} bytes")
            if self.storage:
                lines.append(f"Storage : {self.storage}")
        return lines


def _read_check_tiff(label: Label, deep: bool, block_rows: int = 2048) -> tuple[str, str | None]:
    """
    Return (storage, error). Opens the TIFF with rasterio, compares the
    dimensions with the label and reads the last line (or every line when
    `deep`), so truncation is detected whether or not the file is compressed.
    """
    try:
        import rasterio
        from rasterio.windows import Window
        import warnings
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    except ImportError:
        return "tiff-unknown", (
            "cannot verify TIFF: rasterio is not installed in this Python "
            "(use the project .venv or `conda install -c conda-forge rasterio`)"
        )

    try:
        with rasterio.open(label.payload) as ds:
            compressed = bool(ds.compression)
            storage = "tiff-compressed" if compressed else "tiff-uncompressed"

            if (ds.width, ds.height) != (label.samples, label.lines):
                return storage, (
                    f"TIFF is {ds.width:,} x {ds.height:,} but the label says "
                    f"{label.samples:,} x {label.lines:,}"
                )

            step = block_rows if deep else 1
            starts = range(0, ds.height, step) if deep else [ds.height - 1]
            for y in starts:
                rows = min(step, ds.height - y)
                try:
                    ds.read(1, window=Window(0, y, ds.width, rows))
                except Exception as error:  # rasterio.errors.RasterioIOError etc.
                    return storage, (
                        f"unreadable pixel data starting at line {y:,} "
                        f"of {ds.height:,} ({type(error).__name__})"
                    )
            return storage, None
    except Exception as error:
        return "tiff-unknown", f"cannot open TIFF ({type(error).__name__}: {error})"


def _read_check_raw(label: Label, chunk: int = 64 * 1024 * 1024) -> str | None:
    try:
        with open(label.payload, "rb") as handle:
            while handle.read(chunk):
                pass
    except OSError as error:
        return f"I/O error while reading payload: {error}"
    return None


def validate_product(name: str, deep: bool = False) -> ProductStatus:
    """
    Validate one registered product. Never raises for data problems; the
    result says exactly what is wrong.

    The byte-size rule applies to every uncompressed payload, including
    uncompressed TIFFs (the TMC .tif is one: its size must be >= the pixel
    bytes). Only a *compressed* TIFF is exempt from it; it is checked by
    reading its pixels instead.
    """
    product = PRODUCTS[name]
    try:
        label = parse_label(product.label)
    except (FileNotFoundError, ValueError) as error:
        return ProductStatus(name, False, str(error))

    if not label.payload.is_file():
        return ProductStatus(
            name, False, f"Payload referenced by label is missing: {label.payload}", label
        )

    actual = label.payload.stat().st_size
    is_tiff = label.payload.suffix.lower() in {".tif", ".tiff"}

    if is_tiff:
        storage, error = _read_check_tiff(label, deep)
    else:
        storage, error = "raw", None

    status = ProductStatus(name, True, "payload complete", label, actual, storage)

    if storage != "tiff-compressed" and actual < label.expected_bytes:
        percent = 100.0 * actual / label.expected_bytes
        status.ok = False
        status.message = (
            f"Truncated payload: {actual:,} of {label.expected_bytes:,} bytes "
            f"present ({percent:.1f}%)"
        )
        return status

    if error is None and not is_tiff:
        error = _read_check_raw(label) if deep else None

    if error is not None:
        status.ok = False
        status.message = error
    return status


def validate_products(names, deep: bool = False) -> list[ProductStatus]:
    return [validate_product(name, deep=deep) for name in names]


def print_status(status: ProductStatus) -> None:
    print(f"\n{status.name}")
    print("-" * 70)
    for line in status.describe():
        print(line)
    if status.ok:
        print("Status  : PASS")
    else:
        print(f"Status  : FAIL - {status.message}")


def require_valid(names, deep: bool = False) -> list[ProductStatus]:
    """Validate the given products or exit with status 2 and a clear message."""
    statuses = validate_products(names, deep=deep)
    failed = [s for s in statuses if not s.ok]
    if failed:
        print("\nREQUIRED PRODUCT VALIDATION FAILED")
        print("=" * 70)
        for status in failed:
            print_status(status)
        print(
            "\nThe final OHRC-TMC pair cannot be built from incomplete input. "
            "Re-download / `git lfs pull` the failed product(s) and re-run."
        )
        sys.exit(2)
    return statuses
