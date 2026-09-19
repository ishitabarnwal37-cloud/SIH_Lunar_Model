"""
chandrayaan_dataset_loader.py
================================

Loads and preprocesses REAL Chandrayaan-2 products from the dataset shared
at https://github.com/ankitadeylaik/Ankita-Dey-Laik (forked from
ishitabarnwal37-cloud/SIH_Lunar_Model).

I cloned that repo to inspect it directly. Here's exactly what's in it and
what this script does about each:

  ch2_ohr_ncp_..._d_img_d18/   (OHRC, calibrated)
      data/calibrated/<date>/*.img   <- headerless raw binary, 938 MB
                              *.xml  <- PDS4 label (shape, dtype, geometry)
  ch2_tmc_ndn_..._d_oth_d18/   (TMC-2, derived/ortho)
      data/derived/<date>/*.tif      <- GeoTIFF, 3.5 GB
                          *.xml      <- PDS4 label (redundant w/ TIFF header)
  ch2_iir_nri_..._d_img_d18/   (IIRS, raw)
      data/raw/<date>/*.qub          <- ENVI-style band-sequential cube
                      *.hdr          <- ENVI header (samples/lines/bands/dtype)
                      *.xml          <- PDS4 label (adds per-band wavelengths)

IMPORTANT -- READ THIS BEFORE RUNNING:
The repo tracks *.img, *.tif and *.qub with Git LFS (see its .gitattributes).
When I cloned it, the .img (OHRC) and .tif (TMC) files came down as tiny
~130-byte LFS *pointer* files, not the actual pixel data -- GitHub doesn't
serve LFS binaries through the normal git protocol. The .qub (IIRS) file
DID come through with real bytes, but only ~95 MB out of the true 1.67 GB
(GitHub's 100 MB per-file hard cap for a non-LFS blob) -- so it's a real
but heavily truncated slice of the actual cube (per the ENVI header, only
the first ~15 of 256 bands are actually present).

So, before this script can do anything with OHRC or TMC:
    cd <your clone of the repo>
    git lfs install
    git lfs pull
This downloads the real .img/.tif binaries. Until then, this script will
detect the LFS pointer files, print exactly that, and skip them rather
than silently producing garbage. It DOES already work today on the IIRS
.qub, since real (if partial) bytes are present.

Dependencies: numpy, opencv-python (both already used elsewhere in this
project's other two modules).
"""

from __future__ import annotations

import os
import re
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import cv2

from lunar_preprocessing import (
    LunarPreprocessor,
    select_band_by_wavelength,
    normalize_to_uint8,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("chandrayaan_dataset_loader")

PDS_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

# PDS4 Element_Array/data_type -> numpy dtype string.
# (Covers every value actually used across ISSDC's Chandrayaan-2 archive;
# extend this if a product uses one not listed here -- it will raise a
# clear KeyError rather than silently misreading the bytes.)
PDS4_DTYPE_MAP = {
    "UnsignedByte": "u1",
    "SignedByte": "i1",
    "UnsignedLSB2": "<u2",
    "UnsignedMSB2": ">u2",
    "SignedLSB2": "<i2",
    "SignedMSB2": ">i2",
    "UnsignedLSB4": "<u4",
    "UnsignedMSB4": ">u4",
    "SignedLSB4": "<i4",
    "SignedMSB4": ">i4",
    "IEEE754LSBSingle": "<f4",
    "IEEE754MSBSingle": ">f4",
}


def _is_lfs_pointer(path: str) -> bool:
    """Detect a Git LFS pointer stub instead of real binary data."""
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


# --------------------------------------------------------------------------
# PDS4 XML LABEL PARSING (shared by OHRC's .img and, for metadata, TMC's .tif)
# --------------------------------------------------------------------------

@dataclass
class Pds4ImageInfo:
    file_name: str
    data_type: str
    axes: Dict[str, int]           # e.g. {"Line": 78175, "Sample": 12000}
    offset_bytes: int
    mission_params: Dict[str, str] = field(default_factory=dict)


def parse_pds4_label(xml_path: str) -> Pds4ImageInfo:
    """
    Parse a Chandrayaan-2 PDS4 .xml label and pull out exactly what's needed
    to read the paired raw binary: dtype, axis sizes, byte offset -- plus
    whatever mission/geometry parameters (pixel resolution, sun angles,
    projection, corner coordinates) are present, since your DL/backend
    teammates will likely want those even when you don't.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    file_el = root.find(".//pds:File_Area_Observational/pds:File", PDS_NS)
    file_name = file_el.findtext("pds:file_name", namespaces=PDS_NS)

    array_el = root.find(".//pds:File_Area_Observational/pds:Array_2D_Image", PDS_NS)
    if array_el is None:
        array_el = root.find(".//pds:File_Area_Observational/pds:Array_3D_Spectrum", PDS_NS)
    if array_el is None:
        raise ValueError(f"No Array_2D_Image or Array_3D_Spectrum found in {xml_path}")

    offset_bytes = int(array_el.findtext("pds:offset", default="0", namespaces=PDS_NS))
    data_type = array_el.findtext(".//pds:Element_Array/pds:data_type", namespaces=PDS_NS)

    axes: Dict[str, int] = {}
    for axis_el in array_el.findall("pds:Axis_Array", PDS_NS):
        name = axis_el.findtext("pds:axis_name", namespaces=PDS_NS)
        elements = int(axis_el.findtext("pds:elements", namespaces=PDS_NS))
        axes[name] = elements

    # Best-effort grab of the isda:Product_Parameters (namespace-agnostic,
    # since these are mission-specific extension elements) -- useful
    # metadata to hand off even when we don't act on it here.
    mission_params: Dict[str, str] = {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]  # strip namespace
        if tag in (
            "pixel_resolution", "sun_azimuth", "sun_elevation", "solar_incidence",
            "projection", "processing_level",
        ) and el.text:
            mission_params[tag] = el.text.strip()

    return Pds4ImageInfo(
        file_name=file_name, data_type=data_type, axes=axes,
        offset_bytes=offset_bytes, mission_params=mission_params,
    )


def load_pds4_raw_image(img_path: str, xml_path: str) -> Tuple[np.ndarray, Pds4ImageInfo]:
    """
    Read a headerless PDS4 raw binary (like OHRC's .img) using its paired
    .xml label for shape and dtype. Handles the "Last Index Fastest" axis
    order PDS4 uses, which lines up with numpy's default row-major reshape.
    """
    if _is_lfs_pointer(img_path):
        raise FileNotFoundError(
            f"{img_path} is a Git LFS pointer, not real image data. "
            f"Run `git lfs pull` in your clone of the dataset repo first."
        )

    info = parse_pds4_label(xml_path)
    if info.data_type not in PDS4_DTYPE_MAP:
        raise ValueError(f"Unrecognized PDS4 data_type {info.data_type!r}; "
                          f"add it to PDS4_DTYPE_MAP.")
    dtype = np.dtype(PDS4_DTYPE_MAP[info.data_type])

    lines = info.axes["Line"]
    samples = info.axes["Sample"]
    expected_bytes = lines * samples * dtype.itemsize

    actual_bytes = os.path.getsize(img_path) - info.offset_bytes
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"{img_path} is smaller than the label promises "
            f"({actual_bytes} < {expected_bytes} bytes) -- likely truncated "
            f"or the wrong label is paired with this file."
        )

    with open(img_path, "rb") as f:
        f.seek(info.offset_bytes)
        raw = np.frombuffer(f.read(expected_bytes), dtype=dtype)
    image = raw.reshape(lines, samples)

    logger.info("Loaded PDS4 raw image %s: shape=%s dtype=%s (pixel_resolution=%s m/px)",
                os.path.basename(img_path), image.shape, dtype,
                info.mission_params.get("pixel_resolution", "unknown"))
    return image, info


# --------------------------------------------------------------------------
# ENVI (.hdr + .qub) PARSING -- for IIRS
# --------------------------------------------------------------------------

@dataclass
class EnviCubeInfo:
    samples: int
    lines: int
    bands: int
    dtype: np.dtype
    interleave: str
    wavelengths_nm: Optional[np.ndarray] = None


ENVI_DTYPE_MAP = {
    "1": "u1", "2": "<i2", "3": "<i4", "4": "<f4", "5": "<f8",
    "12": "<u2", "13": "<u4",
}


def parse_envi_header(hdr_path: str) -> EnviCubeInfo:
    """Parse a plain-text ENVI .hdr file into samples/lines/bands/dtype/interleave."""
    fields: Dict[str, str] = {}
    with open(hdr_path) as f:
        for line in f:
            if "=" in line:
                key, val = line.split("=", 1)
                fields[key.strip().lower()] = val.strip()

    byte_order = fields.get("byte order", "0")
    dtype_code = fields.get("data type")
    if dtype_code not in ENVI_DTYPE_MAP:
        raise ValueError(f"Unrecognized ENVI data type code {dtype_code!r} in {hdr_path}")
    dtype_str = ENVI_DTYPE_MAP[dtype_code]
    if dtype_str.startswith("<") and byte_order == "1":
        dtype_str = ">" + dtype_str[1:]  # 1 = big-endian (network order) per ENVI spec

    return EnviCubeInfo(
        samples=int(fields["samples"]),
        lines=int(fields["lines"]),
        bands=int(fields["bands"]),
        dtype=np.dtype(dtype_str),
        interleave=fields.get("interleave", "bsq").lower(),
    )


def parse_iirs_wavelengths_from_xml(xml_path: str) -> np.ndarray:
    """
    IIRS's .hdr doesn't carry per-band wavelengths -- they live in the PDS4
    .xml label's Band_Bin entries instead. Returns an array of center
    wavelengths in micrometers, ordered by band_number.
    """
    xml_text = open(xml_path).read()
    pairs = re.findall(
        r"<band_number>(\d+)</band_number>\s*"
        r"(?:<band_width[^>]*>[\d.]+</band_width>\s*)?"
        r"<center_wavelength unit=\"nm\">([\d.]+)</center_wavelength>",
        xml_text,
    )
    if not pairs:
        raise ValueError(f"No Band_Bin wavelength entries found in {xml_path}")
    pairs.sort(key=lambda p: int(p[0]))
    wavelengths_nm = np.array([float(w) for _, w in pairs], dtype=np.float64)
    return wavelengths_nm / 1000.0  # nm -> um


def load_iirs_band(
    qub_path: str,
    hdr_path: str,
    xml_path: Optional[str] = None,
    target_wavelength_um: float = 1.5,
) -> Tuple[np.ndarray, float]:
    """
    Load the single IIRS band closest to `target_wavelength_um` straight out
    of the .qub cube, without ever materializing the full cube in memory
    (real IIRS cubes are 1.5+ GB).

    Gracefully handles a TRUNCATED .qub (as in the shared repo, where only
    ~95 MB of a 1.67 GB cube is actually present): it computes how many
    complete bands actually exist in the file and, if the target band isn't
    among them, falls back to the closest band that IS present and says so
    loudly rather than reading garbage past end-of-file.

    Returns (band_image, actual_wavelength_used_um).
    """
    if _is_lfs_pointer(qub_path):
        raise FileNotFoundError(
            f"{qub_path} is a Git LFS pointer, not real cube data. "
            f"Run `git lfs pull` in your clone of the dataset repo first."
        )

    info = parse_envi_header(hdr_path)
    if info.interleave != "bsq":
        raise NotImplementedError(
            f"This loader only handles BSQ interleave (got {info.interleave!r}). "
            f"BIL/BIP would need a different band-extraction stride."
        )

    bytes_per_band = info.samples * info.lines * info.dtype.itemsize
    file_size = os.path.getsize(qub_path)
    available_bands = min(info.bands, file_size // bytes_per_band)

    if available_bands == 0:
        raise ValueError(
            f"{qub_path} doesn't even contain one complete band "
            f"({file_size} bytes < {bytes_per_band} bytes/band)."
        )
    if available_bands < info.bands:
        logger.warning(
            "%s is truncated: header describes %d bands but only %d are "
            "fully present (%d of %d bytes). Working with what's available.",
            os.path.basename(qub_path), info.bands, available_bands,
            file_size, info.bands * bytes_per_band,
        )

    wavelengths_um: Optional[np.ndarray] = None
    if xml_path and os.path.exists(xml_path):
        try:
            wavelengths_um = parse_iirs_wavelengths_from_xml(xml_path)
        except ValueError:
            logger.warning("Could not parse wavelengths from %s", xml_path)

    if wavelengths_um is not None:
        available_wavelengths = wavelengths_um[:available_bands]
        band_idx = int(np.argmin(np.abs(available_wavelengths - target_wavelength_um)))
        actual_um = float(available_wavelengths[band_idx])
        if band_idx != int(np.argmin(np.abs(wavelengths_um - target_wavelength_um))):
            logger.warning(
                "True closest-to-%.2fum band isn't in the truncated file; "
                "using the closest AVAILABLE band instead (%.1f nm, index %d).",
                target_wavelength_um, actual_um * 1000, band_idx,
            )
    else:
        logger.warning("No wavelength table available; defaulting to band 0.")
        band_idx = 0
        actual_um = float("nan")

    band_offset = band_idx * bytes_per_band
    with open(qub_path, "rb") as f:
        f.seek(band_offset)
        raw = np.frombuffer(f.read(bytes_per_band), dtype=info.dtype)
    band = raw.reshape(info.lines, info.samples)

    logger.info("Loaded IIRS band %d (%.1f nm) from %s: shape=%s",
                band_idx, actual_um * 1000, os.path.basename(qub_path), band.shape)
    return band, actual_um


# --------------------------------------------------------------------------
# DATASET DISCOVERY (walks the repo's own folder layout)
# --------------------------------------------------------------------------

@dataclass
class DiscoveredProduct:
    sensor: str            # "OHRC", "TMC", or "IIRS"
    product_dir: str
    data_files: Dict[str, str]   # e.g. {"img": ..., "xml": ..., "hdr": ...}


def discover_products(root_dir: str) -> List[DiscoveredProduct]:
    """
    Walk a cloned copy of the dataset repo and find every OHRC/TMC/IIRS
    product folder, matching ISSDC's own naming convention
    (ch2_ohr_*, ch2_tmc_*, ch2_iir_*) rather than hardcoding paths, so this
    also works if more product folders are added later.
    """
    products: List[DiscoveredProduct] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if ".git" in dirpath:
            continue
        exts = {os.path.splitext(fn)[1].lower(): os.path.join(dirpath, fn) for fn in filenames}
        base = os.path.basename(dirpath)

        if any(f.endswith(".img") for f in filenames) and ".xml" in exts:
            sensor = "OHRC" if "ohr" in dirpath.lower() else "UNKNOWN_IMG"
            products.append(DiscoveredProduct(sensor, dirpath, exts))
        elif any(f.endswith(".tif") for f in filenames) and ".xml" in exts:
            sensor = "TMC" if "tmc" in dirpath.lower() else "UNKNOWN_TIF"
            products.append(DiscoveredProduct(sensor, dirpath, exts))
        elif any(f.endswith(".qub") for f in filenames) and ".hdr" in exts:
            sensor = "IIRS" if "iir" in dirpath.lower() else "UNKNOWN_QUB"
            products.append(DiscoveredProduct(sensor, dirpath, exts))

    logger.info("Discovered %d product(s) under %s: %s",
                len(products), root_dir, [p.sensor for p in products])
    return products


# --------------------------------------------------------------------------
# END-TO-END: discover -> load -> preprocess -> save previews
# --------------------------------------------------------------------------

def process_dataset(root_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    products = discover_products(root_dir)

    for product in products:
        try:
            if product.sensor == "OHRC":
                img_path = next(v for k, v in product.data_files.items() if k == ".img")
                xml_path = product.data_files[".xml"]
                raw, info = load_pds4_raw_image(img_path, xml_path)
                # OHRC-NAC recipe (uses src/dst resolution from the label
                # itself where available, else assume 1:1).
                src_res = float(info.mission_params.get("pixel_resolution", 1.0))
                pp = LunarPreprocessor(sensor_pair="OHRC_NAC")
                result = pp.run(raw, src_res_m_per_px=src_res, dst_res_m_per_px=1.0)
                out_path = os.path.join(out_dir, f"{os.path.basename(product.product_dir)}_ohrc.png")
                cv2.imwrite(out_path, result.image)
                logger.info("OHRC preprocessed -> %s", out_path)

            elif product.sensor == "TMC":
                tif_path = next(v for k, v in product.data_files.items() if k == ".tif")
                if _is_lfs_pointer(tif_path):
                    logger.warning("Skipping TMC product at %s: .tif is a Git LFS "
                                    "pointer. Run `git lfs pull` to get the real file.",
                                    product.product_dir)
                    continue
                from lunar_preprocessing import load_image, preprocess_tmc
                raw, _meta = load_image(tif_path)
                band = preprocess_tmc(raw)
                out_path = os.path.join(out_dir, f"{os.path.basename(product.product_dir)}_tmc.png")
                cv2.imwrite(out_path, band)
                logger.info("TMC preprocessed -> %s", out_path)

            elif product.sensor == "IIRS":
                qub_path = next(v for k, v in product.data_files.items() if k == ".qub")
                hdr_path = product.data_files[".hdr"]
                xml_path = product.data_files.get(".xml")
                if _is_lfs_pointer(qub_path):
                    logger.warning("Skipping IIRS product at %s: .qub is a Git LFS "
                                    "pointer. Run `git lfs pull` to get the real file.",
                                    product.product_dir)
                    continue
                band, actual_um = load_iirs_band(qub_path, hdr_path, xml_path)
                band_8bit = normalize_to_uint8(band)
                out_path = os.path.join(out_dir, f"{os.path.basename(product.product_dir)}_iirs_{actual_um:.2f}um.png")
                cv2.imwrite(out_path, band_8bit)
                logger.info("IIRS band preprocessed -> %s", out_path)

            else:
                logger.info("Skipping unrecognized product at %s", product.product_dir)

        except Exception as exc:  # noqa: BLE001 -- report and keep going for other products
            logger.error("Failed to process %s (%s): %s", product.product_dir, product.sensor, exc)


if __name__ == "__main__":
    # Point this at wherever you cloned the dataset repo, e.g.:
    #   git clone https://github.com/ankitadeylaik/Ankita-Dey-Laik.git
    #   git -C Ankita-Dey-Laik lfs install && git -C Ankita-Dey-Laik lfs pull
    #   python3 chandrayaan_dataset_loader.py
    # or override with: CHANDRAYAAN_DATASET_ROOT=/path/to/clone python3 chandrayaan_dataset_loader.py
    _here = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.environ.get("CHANDRAYAAN_DATASET_ROOT", os.path.join(_here, "Ankita-Dey-Laik"))
    OUT_DIR = os.path.join(_here, "outputs", "real_dataset")
    if not os.path.isdir(REPO_ROOT):
        logger.error(
            "Dataset folder not found at %s. Clone it first:\n"
            "    git clone https://github.com/ankitadeylaik/Ankita-Dey-Laik.git\n"
            "or set CHANDRAYAAN_DATASET_ROOT to point at your existing clone.",
            REPO_ROOT,
        )
    else:
        process_dataset(REPO_ROOT, OUT_DIR)
