"""
Paired OHRC-or-NAC / TMC patch dataset for KAN-LoFTR fine-tuning.

Reads a patches .npz with
    ohrc | nac | img0 : (N, 1, P, P) uint8 or float32   image 0 (high resolution)
    tmc  | img1       : (N, 1, P, P) uint8 or float32   image 1 (TMC-2)
    mask              : (N, 1, P, P) optional, 1 = both images valid
    origin_row_col    : (N, 2) patch top-left (row, col) in its strip
    strip_id          : (N,) optional, patches from different strips never interact
and yields, per patch,
    image0, image1  float32 (1, P, P) in [0, 1]      mask0, mask1  float32 (1, P, P)
    H_gt            float32 (3, 3): pixel (x, y) of image0 -> pixel of image1 (exact)

image1 is warped on the fly by a random homography whose 4 corners are
displaced by U(-max_perturb, max_perturb) px, so H_gt is exact GROUND TRUTH
RELATIVE TO THE STORED PAIR. If the stored pair is itself misaligned (the
current OHRC/TMC pairs are only ASSUMED aligned), H_gt is exact only up to
that residual.

Splits are assigned here from origin_row_col, with a guard band: any lower
priority patch (train < val < test) whose L-infinity distance to a patch of
another split is under `guard_px` is dropped, so no ground area leaks across.

    python src/dataset.py --check outputs/pairs/pairs.npz
    python src/dataset.py --convert outputs/pairs/pairs.npz outputs/pairs/patches_u8.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _Base
except ImportError:                       # numpy helpers stay usable without torch
    torch = None
    _Base = object

IMG0_KEYS = ("ohrc", "nac", "img0")
IMG1_KEYS = ("tmc", "img1")
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
STRIDE = 8                                # LoFTR coarse stride: patch size must be a multiple of it


def find_key(data: dict, keys, what: str) -> str:
    for k in keys:
        if k in data:
            return k
    raise KeyError(f"npz has none of {keys} for {what}; found {sorted(data)}")


def to_float01(a: np.ndarray) -> np.ndarray:
    """uint8 -> float32 / 255; float input must already be ~[0, 1]."""
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    a = a.astype(np.float32, copy=False)
    if a.min() < -1e-3 or a.max() > 1.0 + 1e-3:
        raise ValueError(f"float patches must be in [0, 1], got [{a.min():.3f}, {a.max():.3f}]")
    return a


def assign_splits(origin_rc, patch: int, guard_px: int | None = None, block_rows: int | None = None,
                  strip_id=None) -> np.ndarray:
    """
    Split id per patch: 0 train, 1 val, 2 test, -1 dropped by the guard band.

    Along-track blocks of `block_rows` are dealt out (block % 8 == 3 -> val,
    == 7 -> test, else train). Then every patch closer than `guard_px`
    (L-infinity gap between the two P x P squares) to a higher priority patch
    of another split is dropped. Default guard = one patch size.
    """
    origin = np.asarray(origin_rc, dtype=np.int64).reshape(-1, 2)
    n = len(origin)
    guard = patch if guard_px is None else guard_px
    block_rows = block_rows or max(4 * patch, 512)
    sid = np.zeros(n, np.int64) if strip_id is None else np.asarray(strip_id, np.int64).reshape(-1)
    if len(sid) != n:
        raise ValueError("strip_id length does not match origin_row_col")

    block = origin[:, 0] // block_rows
    split = np.zeros(n, np.int64)
    split[block % 8 == 3] = SPLIT_IDS["val"]
    split[block % 8 == 7] = SPLIT_IDS["test"]

    keep = np.ones(n, bool)
    for lo_name, lo in (("train", 0), ("val", 1)):
        lo_idx = np.flatnonzero(split == lo)
        hi_idx = np.flatnonzero(split > lo)
        if len(lo_idx) == 0 or len(hi_idx) == 0:
            continue
        gy = np.abs(origin[lo_idx, None, 0] - origin[None, hi_idx, 0]) - patch
        gx = np.abs(origin[lo_idx, None, 1] - origin[None, hi_idx, 1]) - patch
        gap = np.maximum(np.maximum(gy, gx), 0)
        near = (gap < guard) & (sid[lo_idx, None] == sid[None, hi_idx])
        keep[lo_idx[near.any(1)]] = False
    split[~keep] = -1
    return split


def random_homography(patch: int, max_perturb: float, rng: np.random.Generator) -> np.ndarray:
    """3x3 H (float64) mapping image0 pixels to warped-image1 pixels; corners moved by U(-m, m)."""
    if max_perturb >= patch / 4:
        raise ValueError(f"max_perturb {max_perturb} is too large for {patch}px patches (need < patch/4)")
    src = np.array([[0, 0], [patch - 1, 0], [patch - 1, patch - 1], [0, patch - 1]], np.float32)
    dst = src + rng.uniform(-max_perturb, max_perturb, size=(4, 2)).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def warp_image(img: np.ndarray, H: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    """dst(H x) = src(x) for pixel (x, y) coordinates; invalid border = 0."""
    return cv2.warpPerspective(img, H, (size, size),
                               flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class PairDataset(_Base):
    def __init__(self, npz_path, split: str = "train", warp: bool = True, max_perturb: float = 16.0,
                 photometric: float = 0.5, deterministic: bool = False, seed: int = 0,
                 guard_px: int | None = None, block_rows: int | None = None, repeat: int = 1):
        if split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {list(SPLIT_IDS)}")
        with np.load(Path(npz_path)) as z:
            data = {k: z[k] for k in z.files}
        self.k0 = find_key(data, IMG0_KEYS, "image 0")
        self.k1 = find_key(data, IMG1_KEYS, "image 1")
        self.a0, self.a1 = data[self.k0], data[self.k1]
        if self.a0.shape != self.a1.shape or self.a0.ndim != 4 or self.a0.shape[1] != 1:
            raise ValueError(f"image arrays must both be (N, 1, P, P); got {self.a0.shape} and {self.a1.shape}")
        self.patch = int(self.a0.shape[-1])
        if self.a0.shape[-2] != self.patch or self.patch % STRIDE:
            raise ValueError(f"patches must be square with a side divisible by {STRIDE}, got {self.a0.shape[-2:]}")
        self.mask = data.get("mask")
        if "origin_row_col" in data:
            origin = data["origin_row_col"][:, :2]
        elif "origin_row_col_lat_lon" in data:
            origin = data["origin_row_col_lat_lon"][:, :2]
        else:
            raise KeyError("npz needs origin_row_col: the guard-band split depends on patch positions")
        splits = assign_splits(origin, self.patch, guard_px, block_rows, data.get("strip_id"))
        self.split_counts = {name: int((splits == v).sum()) for name, v in SPLIT_IDS.items()}
        self.n_dropped_by_guard = int((splits < 0).sum())
        self.indices = np.flatnonzero(splits == SPLIT_IDS[split])
        if len(self.indices) == 0:
            raise ValueError(f"split '{split}' is empty (counts {self.split_counts}, {self.n_dropped_by_guard} "
                             "dropped by the guard band); use more patches or a smaller guard_px")
        self.warp, self.max_perturb, self.photometric = warp, max_perturb, photometric
        self.deterministic, self.seed, self.repeat = deterministic, seed, repeat

    def __len__(self) -> int:
        return len(self.indices) * self.repeat

    @staticmethod
    def _jitter(img: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        gamma, contrast = rng.uniform(0.8, 1.25), rng.uniform(0.85, 1.15)
        out = np.clip(0.5 + contrast * (np.power(np.clip(img, 0, 1), gamma) - 0.5), 0, 1)
        return (out * valid).astype(np.float32)

    def get_numpy(self, i: int) -> dict:
        idx = int(self.indices[i % len(self.indices)])
        rng = np.random.default_rng(self.seed + i) if self.deterministic else np.random.default_rng()
        P = self.patch
        img0, img1 = to_float01(self.a0[idx, 0]), to_float01(self.a1[idx, 0])
        valid = np.ones((P, P), np.uint8) if self.mask is None else (self.mask[idx, 0] > 0).astype(np.uint8)
        H = np.eye(3)
        if self.warp:
            H = random_homography(P, self.max_perturb, rng)
            img1 = warp_image(img1, H, P)
            valid1 = warp_image(valid, H, P, nearest=True)
        else:
            valid1 = valid
        if self.photometric > 0:
            if rng.random() < self.photometric:
                img0 = self._jitter(img0, valid, rng)
            if rng.random() < self.photometric:
                img1 = self._jitter(img1, valid1, rng)
        return {"image0": img0[None] * valid[None], "image1": img1[None] * valid1[None],
                "mask0": valid[None].astype(np.float32), "mask1": valid1[None].astype(np.float32),
                "H_gt": H.astype(np.float32), "index": np.int64(idx)}

    def __getitem__(self, i: int) -> dict:
        if torch is None:
            raise ImportError("torch is required to iterate PairDataset (pip install torch)")
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in self.get_numpy(i).items()}


def convert_to_uint8(src: Path, dst: Path) -> None:
    """float32 [0, 1] patches -> compressed uint8 .npz (4x smaller); other arrays are copied."""
    with np.load(src) as z:
        data = {k: z[k] for k in z.files}
    for keys in (IMG0_KEYS, IMG1_KEYS):
        k = find_key(data, keys, "image")
        data[k] = np.round(to_float01(data[k]) * 255).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **data)
    print(f"{src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB: {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", type=Path, help="print split sizes and a sample's shapes")
    ap.add_argument("--convert", nargs=2, type=Path, metavar=("SRC", "DST"), help="float32 npz -> uint8 npz")
    args = ap.parse_args()
    if args.convert:
        convert_to_uint8(*args.convert)
    if args.check:
        for s in SPLIT_IDS:
            try:
                ds = PairDataset(args.check, s)
                print(f"{s:5s}: {len(ds)} patches | counts {ds.split_counts} | dropped by guard {ds.n_dropped_by_guard}")
                if s == "train":
                    sample = ds.get_numpy(0)
                    print("       sample:", {k: getattr(v, "shape", v) for k, v in sample.items()})
            except ValueError as e:
                print(f"{s:5s}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
