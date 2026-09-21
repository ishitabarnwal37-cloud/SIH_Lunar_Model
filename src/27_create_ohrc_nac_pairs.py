#!/usr/bin/env python3
"""Create geometry-supervised OHRC/NAC image pairs for feature matching.

Both inputs must already be georeferenced, map-projected rasters.  The fixed
NAC raster defines the output grid.  GDAL lazily warps the moving OHRC raster
onto that grid, after which valid common tiles are written as grayscale PNGs.

Optional affine perturbations are applied to the moving tile.  Every sample
stores the exact fixed-to-moving and moving-to-fixed 3x3 transforms, so the
result can supervise LoFTR or another correspondence model without using
model-predicted matches as labels.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal


gdal.UseExceptions()


@dataclass(frozen=True)
class Grid:
    projection: str
    geotransform: tuple[float, float, float, float, float, float]
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        gt = self.geotransform
        min_x = gt[0]
        max_y = gt[3]
        max_x = min_x + self.width * gt[1]
        min_y = max_y + self.height * gt[5]
        return min_x, min_y, max_x, max_y


def open_projected_raster(path: Path, role: str) -> gdal.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"{role} raster does not exist: {path}")

    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None or dataset.RasterCount < 1:
        raise RuntimeError(f"GDAL could not read {role} raster: {path}")

    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform(can_return_null=True)
    if not projection or geotransform is None:
        raise RuntimeError(
            f"{role} raster is not map projected: {path}. "
            "Process its camera/geolocation geometry before creating pairs."
        )

    if abs(geotransform[2]) > 1e-12 or abs(geotransform[4]) > 1e-12:
        raise RuntimeError(
            f"{role} raster uses a rotated grid. Warp it to a north-up lunar "
            "grid before creating pairs."
        )
    if geotransform[1] <= 0 or geotransform[5] >= 0:
        raise RuntimeError(f"Unexpected {role} pixel orientation: {geotransform}")

    return dataset


def fixed_grid(dataset: gdal.Dataset) -> Grid:
    return Grid(
        projection=dataset.GetProjection(),
        geotransform=tuple(dataset.GetGeoTransform()),
        width=dataset.RasterXSize,
        height=dataset.RasterYSize,
    )


def warp_moving_to_fixed(moving: gdal.Dataset, grid: Grid) -> gdal.Dataset:
    min_x, min_y, max_x, max_y = grid.bounds
    warped = gdal.Warp(
        "",
        moving,
        format="VRT",
        outputBounds=(min_x, min_y, max_x, max_y),
        width=grid.width,
        height=grid.height,
        dstSRS=grid.projection,
        resampleAlg=gdal.GRA_Average,
        dstAlpha=True,
        multithread=True,
    )
    if warped is None:
        raise RuntimeError("GDAL could not warp the moving raster to the NAC grid.")
    if warped.RasterCount < 2:
        raise RuntimeError("Warped moving raster does not contain a validity band.")
    return warped


def read_fixed_tile(
    dataset: gdal.Dataset, x: int, y: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    band = dataset.GetRasterBand(1)
    image = band.ReadAsArray(x, y, size, size).astype(np.float32)
    mask = band.GetMaskBand().ReadAsArray(x, y, size, size) > 0
    mask &= np.isfinite(image)
    return image, mask


def read_moving_tile(
    dataset: gdal.Dataset, x: int, y: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    image = dataset.GetRasterBand(1).ReadAsArray(x, y, size, size).astype(np.float32)
    alpha = dataset.GetRasterBand(dataset.RasterCount).ReadAsArray(x, y, size, size)
    mask = (alpha > 0) & np.isfinite(image)
    return image, mask


def normalize_tile(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low, high = np.percentile(values, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    scaled[~mask] = 0.0
    return np.round(scaled * 255.0).astype(np.uint8)


def affine_about_center(
    size: int,
    rng: np.random.Generator,
    max_rotation_deg: float,
    max_translation_px: float,
    min_scale: float,
    max_scale: float,
) -> np.ndarray:
    angle = math.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
    scale = rng.uniform(min_scale, max_scale)
    tx = rng.uniform(-max_translation_px, max_translation_px)
    ty = rng.uniform(-max_translation_px, max_translation_px)
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    center = (size - 1) / 2.0

    translate_to_origin = np.array(
        [[1.0, 0.0, -center], [0.0, 1.0, -center], [0.0, 0.0, 1.0]]
    )
    rotate_scale = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    translate_back = np.array(
        [[1.0, 0.0, center + tx], [0.0, 1.0, center + ty], [0.0, 0.0, 1.0]]
    )
    return translate_back @ rotate_scale @ translate_to_origin


def warp_array(array: np.ndarray, forward: np.ndarray, nearest: bool) -> np.ndarray:
    inverse = np.linalg.inv(forward)
    height, width = array.shape
    output_y, output_x = np.indices((height, width), dtype=np.float64)
    source_x = inverse[0, 0] * output_x + inverse[0, 1] * output_y + inverse[0, 2]
    source_y = inverse[1, 0] * output_x + inverse[1, 1] * output_y + inverse[1, 2]

    if nearest:
        sample_x = np.rint(source_x).astype(np.int64)
        sample_y = np.rint(source_y).astype(np.int64)
        valid = (
            (sample_x >= 0)
            & (sample_x < width)
            & (sample_y >= 0)
            & (sample_y < height)
        )
        output = np.zeros(array.shape, dtype=array.dtype)
        output[valid] = array[sample_y[valid], sample_x[valid]]
        return output

    x0 = np.floor(source_x).astype(np.int64)
    y0 = np.floor(source_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (x1 < width) & (y0 >= 0) & (y1 < height)
    output = np.zeros(array.shape, dtype=np.float64)
    dx = source_x[valid] - x0[valid]
    dy = source_y[valid] - y0[valid]
    output[valid] = (
        array[y0[valid], x0[valid]] * (1.0 - dx) * (1.0 - dy)
        + array[y0[valid], x1[valid]] * dx * (1.0 - dy)
        + array[y1[valid], x0[valid]] * (1.0 - dx) * dy
        + array[y1[valid], x1[valid]] * dx * dy
    )
    return np.clip(np.rint(output), 0, 255).astype(array.dtype)


def save_png(array: np.ndarray, path: Path) -> None:
    if array.dtype != np.uint8:
        raise ValueError("PNG output arrays must be uint8.")
    memory = gdal.GetDriverByName("MEM").Create(
        "", array.shape[1], array.shape[0], 1, gdal.GDT_Byte
    )
    memory.GetRasterBand(1).WriteArray(array)
    output = gdal.GetDriverByName("PNG").CreateCopy(str(path), memory)
    if output is None:
        raise RuntimeError(f"Could not write PNG: {path}")
    output = None
    memory = None


def tile_split(y: int, size: int, height: int, train: float, validation: float) -> str:
    position = (y + size / 2.0) / height
    if position < train:
        return "train"
    if position < train + validation:
        return "validation"
    return "test"


def tile_geographic_bounds(
    grid: Grid, x: int, y: int, size: int
) -> tuple[float, float, float, float]:
    gt = grid.geotransform
    min_x = gt[0] + x * gt[1]
    max_x = gt[0] + (x + size) * gt[1]
    max_y = gt[3] + y * gt[5]
    min_y = gt[3] + (y + size) * gt[5]
    return min_x, min_y, max_x, max_y


def save_sample(
    output: Path,
    sample_id: str,
    split: str,
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    fixed_to_moving: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    sample_dir = output / split / sample_id
    sample_dir.mkdir(parents=True, exist_ok=False)

    save_png(fixed, sample_dir / "fixed.png")
    save_png(moving, sample_dir / "moving.png")
    save_png(fixed_mask.astype(np.uint8) * 255, sample_dir / "fixed_mask.png")
    save_png(moving_mask.astype(np.uint8) * 255, sample_dir / "moving_mask.png")

    moving_to_fixed = np.linalg.inv(fixed_to_moving)
    np.savez_compressed(
        sample_dir / "correspondences.npz",
        fixed_to_moving=fixed_to_moving.astype(np.float64),
        moving_to_fixed=moving_to_fixed.astype(np.float64),
        common_valid_mask=(fixed_mask & moving_mask).astype(np.uint8),
    )
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def create_pairs(args: argparse.Namespace) -> dict[str, Any]:
    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError("tile-size and stride must be positive.")
    if args.augmentations <= 0:
        raise ValueError("augmentations must be at least 1.")
    if args.train_fraction <= 0 or args.validation_fraction < 0:
        raise ValueError("Invalid split fractions.")
    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("train-fraction + validation-fraction must be less than 1.")
    if not 0 <= args.min_valid_fraction <= 1:
        raise ValueError("min-valid-fraction must lie in [0, 1].")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {args.output}. "
            "Choose a new directory to avoid mixing dataset versions."
        )

    fixed_ds = open_projected_raster(args.fixed, "Fixed NAC")
    moving_ds = open_projected_raster(args.moving, "Moving OHRC")
    grid = fixed_grid(fixed_ds)
    if grid.width < args.tile_size or grid.height < args.tile_size:
        raise RuntimeError("Fixed raster is smaller than the requested tile size.")

    warped_moving = warp_moving_to_fixed(moving_ds, grid)
    args.output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    candidate_tiles = 0
    accepted_tiles = 0

    for y in range(0, grid.height - args.tile_size + 1, args.stride):
        for x in range(0, grid.width - args.tile_size + 1, args.stride):
            candidate_tiles += 1
            fixed_raw, fixed_mask = read_fixed_tile(fixed_ds, x, y, args.tile_size)
            moving_raw, moving_mask = read_moving_tile(
                warped_moving, x, y, args.tile_size
            )
            common = fixed_mask & moving_mask
            if float(common.mean()) < args.min_valid_fraction:
                continue
            accepted_tiles += 1

            fixed_image = normalize_tile(fixed_raw, fixed_mask)
            moving_image = normalize_tile(moving_raw, moving_mask)
            split = tile_split(
                y,
                args.tile_size,
                grid.height,
                args.train_fraction,
                args.validation_fraction,
            )

            for augmentation in range(args.augmentations):
                if augmentation == 0:
                    fixed_to_moving = np.eye(3, dtype=np.float64)
                else:
                    fixed_to_moving = affine_about_center(
                        args.tile_size,
                        rng,
                        args.max_rotation_deg,
                        args.max_translation_px,
                        args.min_scale,
                        args.max_scale,
                    )

                augmented_moving = warp_array(
                    moving_image, fixed_to_moving, nearest=False
                )
                augmented_mask = warp_array(
                    moving_mask.astype(np.uint8), fixed_to_moving, nearest=True
                ).astype(bool)
                valid_fraction = float((fixed_mask & augmented_mask).mean())
                if valid_fraction < args.min_valid_fraction:
                    continue

                sample_id = f"pair_{len(records):06d}"
                record = {
                    "id": sample_id,
                    "split": split,
                    "moving_sensor": "Chandrayaan-2 OHRC",
                    "fixed_sensor": "LRO LROC NAC",
                    "moving_source": str(args.moving.resolve()),
                    "fixed_source": str(args.fixed.resolve()),
                    "tile_origin_fixed_pixels": [x, y],
                    "tile_size": args.tile_size,
                    "augmentation_index": augmentation,
                    "common_valid_fraction": valid_fraction,
                    "projected_bounds": list(tile_geographic_bounds(grid, x, y, args.tile_size)),
                    "fixed_to_moving": fixed_to_moving.tolist(),
                    "moving_to_fixed": np.linalg.inv(fixed_to_moving).tolist(),
                }
                save_sample(
                    args.output,
                    sample_id,
                    split,
                    fixed_image,
                    augmented_moving,
                    fixed_mask,
                    augmented_mask,
                    fixed_to_moving,
                    record,
                )
                records.append(record)
                split_counts[split] += 1

    manifest = {
        "format_version": 1,
        "description": "Geometry-supervised OHRC moving / NAC fixed pairs",
        "moving_source": str(args.moving.resolve()),
        "fixed_source": str(args.fixed.resolve()),
        "fixed_projection_wkt": grid.projection,
        "fixed_geotransform": list(grid.geotransform),
        "fixed_grid_shape": [grid.height, grid.width],
        "tile_size": args.tile_size,
        "stride": args.stride,
        "augmentations_requested_per_tile": args.augmentations,
        "candidate_tiles": candidate_tiles,
        "accepted_base_tiles": accepted_tiles,
        "sample_count": len(records),
        "split_counts": split_counts,
        "split_strategy": "contiguous fixed-grid y regions",
        "seed": args.seed,
        "samples": records,
    }
    with (args.output / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with (args.output / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create geometry-supervised OHRC moving / NAC fixed image pairs."
    )
    parser.add_argument("--moving", type=Path, required=True, help="Map-projected OHRC raster")
    parser.add_argument("--fixed", type=Path, required=True, help="Map-projected NAC raster")
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=640)
    parser.add_argument("--augmentations", type=int, default=1)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--max-rotation-deg", type=float, default=12.0)
    parser.add_argument("--max-translation-px", type=float, default=48.0)
    parser.add_argument("--min-scale", type=float, default=0.9)
    parser.add_argument("--max-scale", type=float, default=1.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_pairs(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sample_count": manifest["sample_count"],
        "split_counts": manifest["split_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
