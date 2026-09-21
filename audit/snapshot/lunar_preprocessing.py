"""
lunar_preprocessing.py
=======================

Preprocessing pipeline for the SIH 2026 problem statement:
"Multi-modal, Sun-angle and scale invariant image correspondence using
Chandrayaan-2 optical images (OHRC, TMC-2, IIRS)" registered against
reference imagery (LRO NAC, SELENE/Kaguya).

WHERE THIS FITS IN THE TEAM PIPELINE
-------------------------------------
    Raw ISRO / NASA imagery
            |
            v
    [ THIS MODULE: preprocessing.py ]   <-- your part
            |  (clean, normalized, same-scale image pairs
            |   + a metadata dict describing what was done)
            v
    Deep learning matcher (SuperGlue / SuperPoint-style model)  <-- teammate 1
            |  (keypoints, matches, homography)
            v
    Backend + Frontend (visualization, APIs, storage)           <-- teammate 2 & 3

This module does NOT do feature matching or deep learning. Its only job is
to take raw, messy, heterogeneous lunar images and turn them into clean,
comparable, model-ready arrays -- and to record exactly what transforms
were applied so the rest of the team can trust/replay/undo them.

Design reference: Makharia et al., "Comparative Evaluation of Traditional
and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar
Data" (arXiv:2509.04775) -- Section 4 (Methodology) describes the exact
preprocessing steps implemented below (georeferencing, resolution
resampling, intensity normalization, CLAHE, inversion, dilation, PCA,
histogram matching, shadow normalization, log transform).

Dependencies (all pip-installable):
    numpy, opencv-python, scikit-image, scikit-learn
Optional (for real GeoTIFF metadata / georeferencing):
    rasterio  (pip install rasterio)
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import cv2
from skimage.exposure import match_histograms

try:
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAS_RASTERIO = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("lunar_preprocessing")


# --------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------

def load_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a lunar image from disk.

    Handles plain raster formats (png/jpg/tif without CRS) via OpenCV, and
    GeoTIFFs with real georeferencing via rasterio when available.

    Returns
    -------
    image : np.ndarray
        2D (grayscale/panchromatic) or 3D (bands, H, W) array.
    meta : dict
        Metadata: crs, transform, resolution (if known), dtype, shape.
    """
    meta: Dict[str, Any] = {"source_path": path}

    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff") and _HAS_RASTERIO:
        with rasterio.open(path) as src:
            image = src.read()  # (bands, H, W)
            if image.shape[0] == 1:
                image = image[0]
            meta.update(
                {
                    "crs": str(src.crs),
                    "transform": src.transform,
                    "resolution_m_per_px": (
                        abs(src.transform.a),
                        abs(src.transform.e),
                    ),
                    "bounds": src.bounds,
                    "band_count": src.count,
                }
            )
            logger.info("Loaded GeoTIFF %s via rasterio, shape=%s", path, image.shape)
            return image, meta

    # Fallback: plain image, no CRS info.
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    meta.update({"crs": None, "transform": None, "resolution_m_per_px": None})
    logger.info("Loaded raster %s via OpenCV, shape=%s (no georeference found)", path, image.shape)
    return image, meta


# --------------------------------------------------------------------------
# 2. GEOREFERENCING (Section 4.1.1 of the reference paper)
# --------------------------------------------------------------------------

def reproject_to_match(
    src_path: str,
    ref_path: str,
    out_path: str,
) -> str:
    """
    Reproject `src_path` (e.g. OHRC, Selenographic projection) so its CRS
    matches `ref_path` (e.g. LROC NAC, Equirectangular Moon projection).

    This is only meaningful for real GeoTIFFs with CRS metadata, so it
    requires rasterio. If your ISSDC/PRADAN download does not carry CRS
    metadata, coordinate this step with whoever handles georeferencing on
    the team (often bundled with the backend/data-ingestion side) -- the
    rest of this module works fine on plain arrays either way.
    """
    if not _HAS_RASTERIO:
        raise RuntimeError(
            "rasterio is required for CRS reprojection. Install with "
            "`pip install rasterio`. The rest of the pipeline does not "
            "need this step to run."
        )

    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, ref.crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": ref.crs, "transform": transform, "width": width, "height": height}
        )
        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=ref.crs,
                    resampling=Resampling.bilinear,
                )
    logger.info("Reprojected %s -> %s to match CRS of %s", src_path, out_path, ref_path)
    return out_path


# --------------------------------------------------------------------------
# 3. RESOLUTION RESAMPLING (Section 4.1.2)
# --------------------------------------------------------------------------

def resample_to_resolution(
    image: np.ndarray,
    src_res_m_per_px: float,
    dst_res_m_per_px: float,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """
    Resample `image` so that its ground sampling distance (GSD) matches
    `dst_res_m_per_px`, given its current GSD `src_res_m_per_px`.

    Example: OHRC at 0.3 m/px resampled to match NAC at ~1.0 m/px
    -> scale factor = 0.3 / 1.0 = 0.3 (image shrinks).
    """
    if src_res_m_per_px <= 0 or dst_res_m_per_px <= 0:
        raise ValueError("Resolutions must be positive (meters/pixel).")

    scale = src_res_m_per_px / dst_res_m_per_px
    new_w = max(1, int(round(image.shape[1] * scale)))
    new_h = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    logger.info(
        "Resampled image from %s -> %s (scale=%.4f, %.3f m/px -> %.3f m/px)",
        image.shape, resized.shape, scale, src_res_m_per_px, dst_res_m_per_px,
    )
    return resized


# --------------------------------------------------------------------------
# 4. INTENSITY NORMALIZATION (Section 4.1.3)
# --------------------------------------------------------------------------

def normalize_to_uint8(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """
    Normalize any-dtype image (uint16 DN counts, float radiance, etc.) to
    the standard 8-bit (0-255) range using a percentile stretch, which is
    robust to a few extreme hot/dead pixels (common in raw satellite data).
    """
    img = image.astype(np.float64)
    lo, hi = np.percentile(img, [lower_percentile, upper_percentile])
    if hi <= lo:
        hi = lo + 1e-6
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo) * 255.0
    return img.astype(np.uint8)


# --------------------------------------------------------------------------
# 5. OHRC <-> NAC SPECIALISED STEPS (Section 4.2 A)
# --------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def invert_image(image: np.ndarray) -> np.ndarray:
    """Complement pixel values: highlights features hidden in dark/shadow areas."""
    return 255 - image


def morphological_dilate(
    image: np.ndarray, kernel_size: int = 3, iterations: int = 1
) -> np.ndarray:
    """Expand bright structures (crater rims, ridges) to strengthen edges for keypoints."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel, iterations=iterations)


def pca_denoise(image: np.ndarray, patch_size: int = 8, n_components: float = 0.95) -> np.ndarray:
    """
    Lightweight PCA-based denoising for a single-band image: the image is
    split into small patches, PCA keeps the components explaining
    `n_components` fraction of variance, and patches are reconstructed with
    noise-dominated components removed. Useful as an optional extra
    cleanup step before feature extraction on noisy OHRC crops.
    """
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for pca_denoise. `pip install scikit-learn`.")

    h, w = image.shape
    ph = h - (h % patch_size)
    pw = w - (w % patch_size)
    cropped = image[:ph, :pw].astype(np.float64)

    patches = (
        cropped.reshape(ph // patch_size, patch_size, pw // patch_size, patch_size)
        .swapaxes(1, 2)
        .reshape(-1, patch_size * patch_size)
    )

    pca = PCA(n_components=n_components, svd_solver="full")
    transformed = pca.fit_transform(patches)
    reconstructed = pca.inverse_transform(transformed)

    out = (
        reconstructed.reshape(ph // patch_size, pw // patch_size, patch_size, patch_size)
        .swapaxes(1, 2)
        .reshape(ph, pw)
    )
    out = np.clip(out, 0, 255).astype(np.uint8)

    # paste back onto a full-size canvas (edges beyond the crop keep original values)
    result = image.copy()
    result[:ph, :pw] = out
    return result


# --------------------------------------------------------------------------
# 6. IIRS <-> WAC SPECIALISED STEPS (Section 4.2 B)
# --------------------------------------------------------------------------

def select_reference_band(hyperspectral_cube: np.ndarray, band_index: Optional[int] = None) -> np.ndarray:
    """
    From a hyperspectral cube (bands, H, W), pick one visually clear band to
    use as the reference for registration. Transformation parameters found
    on this band get reused on every other band later (not this module's
    job -- that's the DL/matching stage -- but this is where the single
    band is chosen).
    """
    if hyperspectral_cube.ndim != 3:
        raise ValueError("Expected a (bands, H, W) hyperspectral cube.")
    if band_index is None:
        # Heuristic: pick the band with the highest contrast (std dev).
        stds = hyperspectral_cube.reshape(hyperspectral_cube.shape[0], -1).std(axis=1)
        band_index = int(np.argmax(stds))
        logger.info("Auto-selected band %d as reference (highest contrast).", band_index)
    return hyperspectral_cube[band_index]


def histogram_match_to_reference(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the intensity distribution of `source` to that of `reference`."""
    matched = match_histograms(source.astype(np.float64), reference.astype(np.float64))
    matched = np.clip(matched, 0, 255)
    return matched.astype(np.uint8)


def shadow_normalize(
    image: np.ndarray,
    shadow_thresh: int = 40,
    gain: float = 1.8,
) -> np.ndarray:
    """
    Brighten pixels below `shadow_thresh` to reveal detail lost in lunar
    shadow regions, while leaving already-bright pixels alone. Simple,
    fast, and effective compared to full radiometric shadow modelling.
    """
    img = image.astype(np.float64)
    shadow_mask = img < shadow_thresh
    img[shadow_mask] = np.clip(img[shadow_mask] * gain, 0, 255)
    return img.astype(np.uint8)


def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Compress high-intensity values while boosting low-intensity detail.
    Classic transform for images with a very large dynamic range (e.g.
    hyperspectral / SAR data).
    """
    img = image.astype(np.float64)
    normalized = img / 255.0
    log_img = c * np.log1p(normalized)
    log_img = log_img / log_img.max() * 255.0 if log_img.max() > 0 else log_img
    return log_img.astype(np.uint8)


# --------------------------------------------------------------------------
# 6b. TMC-2 SPECIALISED STEPS
# --------------------------------------------------------------------------
# TMC-2 (Terrain Mapping Camera-2) images the Moon from three look angles
# per orbit -- Fore, Nadir, Aft (this is how it does stereo/DEM generation).
# For cross-registration with OHRC and IIRS, the reference paper's team
# found the Nadir look angle works best (least foreshortening, most
# directly comparable to OHRC/IIRS which are themselves near-nadir).
# The recipe here is intentionally simple: pick the nadir band, then CLAHE.

def select_tmc_nadir_band(
    tmc_data: np.ndarray,
    band_labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Select the Nadir-look band out of a multi-band/multi-file TMC-2 product.

    `tmc_data` can be:
      - a 2D array: already a single band, returned as-is.
      - a 3D array (bands, H, W): if `band_labels` is given (e.g.
        ["Fore", "Nadir", "Aft"], read from the product's .xml/.lbl label
        file), the matching band is picked. Otherwise the middle band is
        used as a heuristic, since ISRO's TMC-2 strips are conventionally
        ordered Fore -> Nadir -> Aft.
    """
    if tmc_data.ndim == 2:
        return tmc_data

    if tmc_data.ndim != 3:
        raise ValueError("Expected a 2D band or a (bands, H, W) TMC-2 stack.")

    if band_labels is not None:
        try:
            idx = [b.lower() for b in band_labels].index("nadir")
        except ValueError:
            logger.warning("No 'Nadir' label found in band_labels; falling back to middle band.")
            idx = tmc_data.shape[0] // 2
    else:
        idx = tmc_data.shape[0] // 2
        logger.info("No band_labels provided; assuming conventional Fore/Nadir/Aft "
                    "ordering and picking the middle band (index %d) as Nadir.", idx)
    return tmc_data[idx]


def preprocess_tmc(
    tmc_data: np.ndarray,
    band_labels: Optional[List[str]] = None,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """TMC-2 recipe: Nadir band selection -> normalize -> CLAHE."""
    band = select_tmc_nadir_band(tmc_data, band_labels)
    band = normalize_to_uint8(band)
    band = apply_clahe(band, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return band


# --------------------------------------------------------------------------
# 6c. IIRS SPECIALISED STEPS (hyperspectral, QUB cube)
# --------------------------------------------------------------------------
# IIRS delivers data as a QUB (ISIS "qube") hyperspectral cube: hundreds of
# narrow spectral bands per pixel. For registration purposes we don't need
# the whole cube -- we need ONE band that shows surface morphology clearly.
# The team's chosen band is the one nearest 1.5 micron: short-wave infrared
# at 1.5 um shows strong albedo/mineral contrast on the lunar surface and
# is largely free of the noisiest bands near IIRS's detector-edge
# wavelengths, making it a good match target against WAC/NAC.

def extract_qub_cube(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load an IIRS QUB (ISIS cube) file into a (bands, H, W) array, plus the
    per-band wavelength table in micrometers if it can be recovered.

    QUB is ISRO/ISIS's cube format -- there isn't a single universal pure-
    Python reader, so this function tries a couple of common routes and
    raises a clear, actionable error if neither is installed rather than
    guessing at file internals:

      1. `spectral` (SPy) -- works if the QUB ships with a matching .hdr
         (ENVI-style header), which many ISSDC/PRADAN IIRS products do.
      2. `pysis` -- for true ISIS cube headers.

    Coordinate with your data-ingestion teammate on which of these your
    actual downloaded files need; once you have a (bands, H, W) numpy
    array + wavelengths, everything below (`select_band_by_wavelength`,
    `preprocess_iirs`) is format-agnostic and works regardless of loader.
    """
    try:
        import spectral  # type: ignore
        img = spectral.open_image(path)
        cube = np.asarray(img.load()).transpose(2, 0, 1)  # (bands, H, W)
        wavelengths = None
        if hasattr(img, "bands") and getattr(img.bands, "centers", None):
            wavelengths = np.array(img.bands.centers, dtype=np.float64)
            if wavelengths.max() > 50:  # heuristic: values in nm, convert to um
                wavelengths = wavelengths / 1000.0
        logger.info("Loaded QUB cube via `spectral`: shape=%s", cube.shape)
        return cube, wavelengths
    except ImportError:
        pass

    try:
        from pysis import CubeFile  # type: ignore
        cube_file = CubeFile.open(path)
        cube = np.asarray(cube_file.data)  # typically (bands, H, W) already
        logger.info("Loaded QUB cube via `pysis`: shape=%s", cube.shape)
        return cube, None  # pysis doesn't expose wavelengths uniformly; supply separately
    except ImportError:
        pass

    raise ImportError(
        "Could not load QUB cube: neither `spectral` nor `pysis` is installed. "
        "Install one with `pip install spectral` (needs an accompanying ENVI "
        ".hdr file) or `pip install pysis` (true ISIS cubes), or ask whoever "
        "handles data ingestion for a pre-extracted (bands, H, W) numpy array."
    )


def select_band_by_wavelength(
    cube: np.ndarray,
    wavelengths_um: np.ndarray,
    target_um: float = 1.5,
) -> Tuple[np.ndarray, float]:
    """
    Pick the cube band whose center wavelength is closest to `target_um`
    micrometers (default: the 1.5 um band the team has settled on).

    Returns (band_2d, actual_wavelength_used_um).
    """
    if cube.ndim != 3:
        raise ValueError("Expected a (bands, H, W) cube.")
    if len(wavelengths_um) != cube.shape[0]:
        raise ValueError(
            f"wavelengths_um has {len(wavelengths_um)} entries but cube has "
            f"{cube.shape[0]} bands -- these must correspond 1:1."
        )
    idx = int(np.argmin(np.abs(np.asarray(wavelengths_um) - target_um)))
    logger.info("Selected IIRS band %d (%.3f um) as closest match to target %.2f um",
                idx, wavelengths_um[idx], target_um)
    return cube[idx], float(wavelengths_um[idx])


def preprocess_iirs(
    cube: np.ndarray,
    wavelengths_um: Optional[np.ndarray] = None,
    target_wavelength_um: float = 1.5,
    reference_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    IIRS recipe: QUB band selection near 1.5 um -> normalize -> (optional)
    histogram match to a WAC/NAC reference -> shadow normalize -> log transform.

    If `wavelengths_um` isn't available (e.g. cube arrived pre-extracted
    without a header), falls back to the generic highest-contrast band
    heuristic used elsewhere in this module, and logs a warning so it's
    obvious in the pipeline metadata that the 1.5 um target wasn't actually
    used.
    """
    if wavelengths_um is not None:
        band, actual_um = select_band_by_wavelength(cube, wavelengths_um, target_wavelength_um)
    else:
        logger.warning(
            "No wavelength table supplied for IIRS cube; falling back to "
            "highest-contrast band instead of the true 1.5 um band. Get the "
            "wavelength table from the QUB header for the real pipeline."
        )
        band = select_reference_band(cube)
        actual_um = None

    band = normalize_to_uint8(band)
    if reference_image is not None:
        band = histogram_match_to_reference(band, normalize_to_uint8(reference_image))
    band = shadow_normalize(band)
    band = log_transform(band)
    return band


# --------------------------------------------------------------------------
# 7. DFSAR <-> SELENE SPECIALISED STEPS (radar-specific)
# --------------------------------------------------------------------------

def despeckle(image: np.ndarray, method: str = "median", ksize: int = 5) -> np.ndarray:
    """
    Suppress SAR speckle noise before feature extraction. `median` is fast
    and edge-preserving-ish; `bilateral` preserves crater/ridge edges better
    but is slower.
    """
    if method == "median":
        return cv2.medianBlur(image, ksize)
    if method == "bilateral":
        return cv2.bilateralFilter(image, d=ksize, sigmaColor=50, sigmaSpace=50)
    raise ValueError(f"Unknown despeckle method: {method}")


# --------------------------------------------------------------------------
# 8. HIGH-LEVEL PIPELINE (ties everything together per sensor pair)
# --------------------------------------------------------------------------

SensorPair = str  # one of "OHRC_NAC", "IIRS_WAC", "DFSAR_SELENE"


@dataclass
class PreprocessResult:
    image: np.ndarray
    steps_applied: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json_meta(self) -> str:
        """Serializable summary for handing off to backend/DL teammates."""
        safe_meta = {k: v for k, v in self.meta.items() if not isinstance(v, np.ndarray)}
        return json.dumps({"steps_applied": self.steps_applied, "meta": safe_meta}, default=str, indent=2)


class LunarPreprocessor:
    """
    Orchestrates the full preprocessing pipeline for a given sensor pair.

    Usage
    -----
        pp = LunarPreprocessor(sensor_pair="OHRC_NAC")
        result = pp.run(raw_image, src_res_m_per_px=0.3, dst_res_m_per_px=1.0)
        clean_image = result.image
        model_input = convert_to_model_input(clean_image)
    """

    VALID_PAIRS = ("OHRC_NAC", "IIRS_WAC", "DFSAR_SELENE", "TMC_NAC", "IIRS_1p5UM")

    def __init__(self, sensor_pair: SensorPair):
        if sensor_pair not in self.VALID_PAIRS:
            raise ValueError(f"sensor_pair must be one of {self.VALID_PAIRS}, got {sensor_pair!r}")
        self.sensor_pair = sensor_pair

    def run(
        self,
        image: np.ndarray,
        src_res_m_per_px: Optional[float] = None,
        dst_res_m_per_px: Optional[float] = None,
        reference_image: Optional[np.ndarray] = None,
        band_labels: Optional[List[str]] = None,
        wavelengths_um: Optional[np.ndarray] = None,
        target_wavelength_um: float = 1.5,
    ) -> PreprocessResult:
        steps: List[str] = []
        img = image

        # The two wavelength/band-aware pairs handle their own band
        # selection internally (they need extra info -- labels or a
        # wavelength table), so they short-circuit the generic path below.
        if self.sensor_pair == "TMC_NAC":
            img = preprocess_tmc(img, band_labels=band_labels)
            steps += ["select_tmc_nadir_band", "normalize_to_uint8", "apply_clahe"]
            meta = {"sensor_pair": self.sensor_pair, "output_shape": img.shape, "output_dtype": str(img.dtype)}
            return PreprocessResult(image=img, steps_applied=steps, meta=meta)

        if self.sensor_pair == "IIRS_1p5UM":
            img = preprocess_iirs(
                img, wavelengths_um=wavelengths_um,
                target_wavelength_um=target_wavelength_um,
                reference_image=reference_image,
            )
            steps += ["select_band_by_wavelength(~1.5um)", "normalize_to_uint8",
                      "histogram_match_to_reference" if reference_image is not None else None,
                      "shadow_normalize", "log_transform"]
            steps = [s for s in steps if s]
            meta = {"sensor_pair": self.sensor_pair, "output_shape": img.shape, "output_dtype": str(img.dtype)}
            return PreprocessResult(image=img, steps_applied=steps, meta=meta)

        # Hyperspectral cube -> pick a reference band first (generic path).
        if img.ndim == 3:
            img = select_reference_band(img)
            steps.append("select_reference_band")

        # Step 1: resolution alignment (common to all pairs).
        if src_res_m_per_px and dst_res_m_per_px:
            img = resample_to_resolution(img, src_res_m_per_px, dst_res_m_per_px)
            steps.append(f"resample_to_resolution({src_res_m_per_px}->{dst_res_m_per_px} m/px)")

        # Step 2: normalize to 8-bit (common to all pairs).
        img = normalize_to_uint8(img)
        steps.append("normalize_to_uint8")

        # Step 3: sensor-pair-specific enhancement.
        if self.sensor_pair == "OHRC_NAC":
            img = apply_clahe(img)
            steps.append("apply_clahe")
            img = invert_image(img)
            steps.append("invert_image")
            img = morphological_dilate(img)
            steps.append("morphological_dilate")

        elif self.sensor_pair == "IIRS_WAC":
            if reference_image is not None:
                ref = normalize_to_uint8(reference_image)
                img = histogram_match_to_reference(img, ref)
                steps.append("histogram_match_to_reference")
            img = shadow_normalize(img)
            steps.append("shadow_normalize")
            img = log_transform(img)
            steps.append("log_transform")

        elif self.sensor_pair == "DFSAR_SELENE":
            img = despeckle(img, method="median")
            steps.append("despeckle(median)")
            img = log_transform(img)
            steps.append("log_transform")

        meta = {
            "sensor_pair": self.sensor_pair,
            "output_shape": img.shape,
            "output_dtype": str(img.dtype),
        }
        logger.info("Pipeline [%s] complete. Steps: %s", self.sensor_pair, steps)
        return PreprocessResult(image=img, steps_applied=steps, meta=meta)


# --------------------------------------------------------------------------
# 9. HANDOFF TO THE DEEP LEARNING MODEL
# --------------------------------------------------------------------------

def convert_to_model_input(
    image: np.ndarray,
    target_size: Tuple[int, int] = (480, 640),
    to_float_tensor: bool = True,
    replicate_channels: bool = False,
) -> np.ndarray:
    """
    Final handoff step: turn a preprocessed uint8 image into whatever your
    teammate's model expects.

    - Resizes to a fixed (H, W) so batches/pairs are shape-consistent.
    - Optionally scales to float32 in [0, 1] (most PyTorch/TF models expect this).
    - Optionally replicates a single channel to 3 channels (for models that
      expect RGB-shaped input even on grayscale data, e.g. ImageNet-pretrained backbones).

    Returns an array shaped (H, W), (1, H, W) or (3, H, W) depending on flags --
    confirm the exact expected shape with whoever owns the model code.
    """
    resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    if to_float_tensor:
        resized = resized.astype(np.float32) / 255.0

    if replicate_channels:
        resized = np.stack([resized] * 3, axis=0)  # (3, H, W)
    else:
        resized = resized[np.newaxis, ...]  # (1, H, W)

    return resized


# --------------------------------------------------------------------------
# 10. DEMO / SELF-TEST
# --------------------------------------------------------------------------

def _make_synthetic_crater_image(
    size: Tuple[int, int] = (600, 600),
    n_craters: int = 40,
    illumination_angle_deg: float = 30.0,
    noise_std: float = 6.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Generates a synthetic 'lunar-like' grayscale surface with circular
    craters that have a bright rim on the illumination side and a dark
    shadow on the opposite side -- standing in for real OHRC/NAC crops so
    this script can be demoed and tested without needing ISRO/NASA data
    on hand.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    base = np.full((h, w), 120, dtype=np.float64)

    theta = np.deg2rad(illumination_angle_deg)
    light_dx, light_dy = np.cos(theta), np.sin(theta)

    for _ in range(n_craters):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(8, 45)
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        crater_mask = dist <= r

        # Shade: bright on the illumination side, dark on the opposite side.
        proj = ((xx - cx) * light_dx + (yy - cy) * light_dy) / (r + 1e-6)
        shade = np.clip(proj, -1, 1) * 70
        base[crater_mask] -= shade[crater_mask]
        rim_mask = (dist > r * 0.85) & (dist <= r)
        base[rim_mask] += 25

    noise = rng.normal(0, noise_std, size=(h, w))
    base += noise
    return np.clip(base, 0, 255)


def _demo() -> None:
    """Runs the full pipeline end-to-end on synthetic data and reports results."""
    logger.info("=== DEMO: synthetic OHRC-like vs NAC-like pair ===")

    # Simulate a high-res OHRC crop (finer detail, more noise) and a lower
    # resolution NAC reference of a nearby, slightly rotated area.
    ohrc_like = _make_synthetic_crater_image(
        size=(900, 900), n_craters=60, illumination_angle_deg=25, noise_std=8, seed=1
    )
    nac_like = _make_synthetic_crater_image(
        size=(400, 400), n_craters=60, illumination_angle_deg=40, noise_std=3, seed=1
    )

    pp = LunarPreprocessor(sensor_pair="OHRC_NAC")
    result = pp.run(ohrc_like, src_res_m_per_px=0.3, dst_res_m_per_px=1.0)

    print("\n--- OHRC_NAC pipeline result ---")
    print(result.to_json_meta())

    model_input = convert_to_model_input(result.image, target_size=(256, 256))
    print(f"Model-ready tensor shape: {model_input.shape}, dtype: {model_input.dtype}, "
          f"range: [{model_input.min():.3f}, {model_input.max():.3f}]")

    # Also exercise the IIRS_WAC branch on a fake 3-band hyperspectral cube.
    logger.info("=== DEMO: synthetic IIRS-like hyperspectral cube vs WAC-like reference ===")
    cube = np.stack(
        [_make_synthetic_crater_image(size=(300, 300), seed=s) for s in range(5)]
    )
    wac_like = _make_synthetic_crater_image(size=(300, 300), seed=99, noise_std=2)

    pp2 = LunarPreprocessor(sensor_pair="IIRS_WAC")
    result2 = pp2.run(cube, reference_image=wac_like)
    print("\n--- IIRS_WAC pipeline result ---")
    print(result2.to_json_meta())

    # Save visual outputs so results can be sanity-checked by eye.
    # Relative path: creates an "outputs" folder next to this script,
    # wherever it's run from -- no dependency on any specific machine.
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "demo_ohrc_raw.png"), ohrc_like.astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "demo_ohrc_preprocessed.png"), result.image)
    cv2.imwrite(os.path.join(out_dir, "demo_nac_reference.png"), nac_like.astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "demo_iirs_band_preprocessed.png"), result2.image)
    logger.info("Saved demo images to %s", out_dir)


if __name__ == "__main__":
    _demo()
