"""
Build a paired, patched OHRC 2024 / TMC dataset from real georeferencing.

Unlike the fixed-corridor scripts (11/13/17/18), this warps OHRC onto the TMC
ortho pixel grid using OHRC's own geometry file (121 x 799 lat/lon control
points), finds where both images really have data, and cuts fixed-size
patches from that overlap only.

    python src/27_build_patch_dataset.py --dry-run
    python src/27_build_patch_dataset.py --offset-east-m 420 --offset-north-m -60

--offset-east-m / --offset-north-m is the registration correction: how far
the true ground position of OHRC lies east / north of its nominal geometry.
Measure it against an independent reference; it is NOT estimated here.

Exit codes: 0 dataset written, 2 invalid input product, 3 no overlap
(nothing is written; the report says why).

Outputs (outputs/patches/): patches.npz, index.json, preview.png
    ohrc, tmc : float32 (N, 1, P, P) in [0, 1]     mask : uint8 (N, 1, P, P)
    split     : 0 train / 1 val (spatially disjoint along-track blocks)
    qa_*      : per-patch alignment diagnostics (informational, see index.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_utils import DATA_ROOT, FINAL_PRODUCTS, OUTPUTS_DIR, PRODUCTS, parse_label, require_valid  # noqa: E402
from raster_io import open_raster  # noqa: E402

M_PER_DEG = 1737400.0 * np.pi / 180.0        # metres per degree at the lunar equator
OHRC_M_PER_PX = 0.30                          # pixel_resolution in the OHRC label


def poly_terms(x, y, degree=3):
    return np.stack([x ** i * y ** j for i in range(degree + 1) for j in range(degree + 1 - i)], -1)


def fit_ohrc_geometry(csv_path: Path):
    """Polynomial (lon, lat) -> (sample, line) fitted to the control grid."""
    lon, lat, sample, line = np.loadtxt(csv_path, delimiter=",", skiprows=1).T
    lon0, lat0 = lon.mean(), lat.mean()
    a = poly_terms((lon - lon0) * 10, (lat - lat0) * 10)
    cs = np.linalg.lstsq(a, sample, rcond=None)[0]
    cl = np.linalg.lstsq(a, line, rcond=None)[0]
    err = max(abs(a @ cs - sample).max(), abs(a @ cl - line).max())
    if err > 3:
        sys.exit(f"ERROR: OHRC geometry fit residual {err:.1f} px is too large.")
    return dict(lon0=lon0, lat0=lat0, cs=cs, cl=cl, lon=lon, lat=lat), err


def block_mean(path, factor):
    """OHRC block-mean downsampled by `factor`, streamed strip by strip."""
    with open_raster(path) as ds:
        h, w = ds.height // factor, ds.width // factor
        out = np.zeros((h, w), np.float32)
        step = factor * 128
        for y in range(0, h * factor, step):
            rows = min(step, h * factor - y)
            a = ds.read(1, window=Window(0, y, w * factor, rows)).astype(np.float32)
            out[y // factor:(y + rows) // factor] = a.reshape(rows // factor, factor, w, factor).mean((1, 3))
    return out


def lcn(a, m, hp=5, n=10):
    """Band-pass + local contrast normalisation (QA only)."""
    m = m.astype(np.float32)
    bl = lambda x, s: cv2.GaussianBlur(x, (0, 0), s)  # noqa: E731
    h = (a - bl(a * m, hp) / (bl(m, hp) + 1e-6)) * m
    sd = np.sqrt(bl(h * h, n) / (bl(m, n) + 1e-6) + 1e-6)
    return (h / sd * m).astype(np.float32)


def stretch(a, mask, lo=1, hi=99):
    v = a[mask]
    p0, p1 = np.percentile(v, (lo, hi))
    return np.clip((a - p0) / max(p1 - p0, 1e-6), 0, 1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offset-east-m", type=float, default=0.0)
    ap.add_argument("--offset-north-m", type=float, default=0.0)
    ap.add_argument("--patch", type=int, default=128, help="patch size in TMC pixels")
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--min-valid", type=float, default=0.90,
                    help="minimum fraction of a patch where both images have data")
    ap.add_argument("--val-every", type=int, default=5, help="every Nth along-track block is validation")
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "patches")
    ap.add_argument("--dry-run", action="store_true", help="only report the overlap")
    args = ap.parse_args()

    require_valid(FINAL_PRODUCTS)
    ohrc_p, tmc_p = PRODUCTS["OHRC_2024"], PRODUCTS["TMC"]
    ohrc_label, tmc_label = parse_label(ohrc_p.label), parse_label(tmc_p.label)
    csv = (DATA_ROOT / ohrc_p.folder / "geometry" / ohrc_p.level / ohrc_p.date
           / f"{ohrc_p.folder.replace('_d_img_d18', '_g_grd_d18')}.csv")
    geo, fit_err = fit_ohrc_geometry(csv)
    print(f"OHRC geometry fit: max residual {fit_err:.2f} OHRC px")

    with open_raster(tmc_label.payload) as tmc_ds:
        t = tmc_ds.transform
        s = t.a
        # OHRC footprint (with the registration offset) in TMC ortho pixels
        lat_c = float(np.mean([geo["lat"].min(), geo["lat"].max()]))
        d_lon = args.offset_east_m / (M_PER_DEG * np.cos(np.radians(lat_c)))
        d_lat = args.offset_north_m / M_PER_DEG
        c0 = int((geo["lon"].min() + d_lon - t.c) / s) - 4
        c1 = int((geo["lon"].max() + d_lon - t.c) / s) + 5
        r0 = int((t.f - (geo["lat"].max() + d_lat)) / -t.e) - 4
        r1 = int((t.f - (geo["lat"].min() + d_lat)) / -t.e) + 5
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, tmc_ds.width), min(r1, tmc_ds.height)
        tmc = tmc_ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)

    factor = max(1, int(round(s * M_PER_DEG / OHRC_M_PER_PX)))
    print(f"TMC ortho pixel {s * M_PER_DEG:.2f} m; OHRC block-mean factor {factor}")
    ohr_small = block_mean(ohrc_label.path, factor)

    rr, cc = np.mgrid[r0:r1, c0:c1]
    lon = t.c + (cc + 0.5) * s - d_lon            # ground -> nominal OHRC coordinates
    lat = t.f + (rr + 0.5) * t.e - d_lat
    a = poly_terms((lon - geo["lon0"]) * 10, (lat - geo["lat0"]) * 10)
    sample, line = a @ geo["cs"], a @ geo["cl"]
    mx = ((sample - (factor - 1) / 2) / factor).astype(np.float32)
    my = ((line - (factor - 1) / 2) / factor).astype(np.float32)
    ohr = cv2.remap(ohr_small, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ohr_ok = (sample >= 0) & (sample <= ohrc_label.samples - 1) & (line >= 0) & (line <= ohrc_label.lines - 1) & (ohr > 0)
    tmc_ok = tmc > 0
    both = ohr_ok & tmc_ok

    print(f"registration offset: east {args.offset_east_m:+.0f} m, north {args.offset_north_m:+.0f} m")
    print(f"OHRC-covered px {int(ohr_ok.sum()):,} | TMC valid px {int(tmc_ok.sum()):,} | overlap {int(both.sum()):,}")
    if both.sum() == 0:
        gap = np.median([np.flatnonzero(tmc_ok[i])[0] - np.flatnonzero(ohr_ok[i])[-1]
                         for i in range(0, both.shape[0], 200) if tmc_ok[i].any() and ohr_ok[i].any()])
        print(f"NO OVERLAP: TMC data starts a median {gap:.0f} px ({gap * s * M_PER_DEG:.0f} m) east of the "
              "OHRC footprint. Supply a registration offset that moves OHRC east by more than this, "
              "or use an OHRC strip that lies inside the TMC swath.")
        return 3
    widths = both.sum(1)
    print(f"overlap width per row: median {np.median(widths[widths > 0]):.0f} px, max {widths.max()} px")
    if args.dry_run:
        return 0

    ohr_n, tmc_n = stretch(ohr, both), stretch(tmc, both)
    lo_, lt_ = lcn(ohr_n, both), lcn(tmc_n, both)
    P, S = args.patch, args.stride
    rows_ok = np.flatnonzero(both.any(1))
    items, coords = [], []
    for y in range(rows_ok[0], rows_ok[-1] - P + 2, S):
        for x in range(0, both.shape[1] - P + 1, S):
            m = both[y:y + P, x:x + P]
            if m.mean() >= args.min_valid:
                items.append((y, x))
    if not items:
        print(f"No {P}x{P} patch reaches {args.min_valid:.0%} overlap (overlap strip too narrow); try --patch 64.")
        return 3

    N = len(items)
    A = np.zeros((N, 1, P, P), np.float32); B = np.zeros_like(A); M = np.zeros((N, 1, P, P), np.uint8)
    ncc0 = np.zeros(N, np.float32); bestncc = np.zeros(N, np.float32); shift = np.zeros((N, 2), np.int16)
    origin = np.zeros((N, 4), np.float64); split = np.zeros(N, np.uint8)
    for i, (y, x) in enumerate(items):
        sl = (slice(y, y + P), slice(x, x + P))
        m = both[sl]
        A[i, 0], B[i, 0], M[i, 0] = ohr_n[sl] * m, tmc_n[sl] * m, m
        a_, b_ = lo_[sl], lt_[sl]
        c = (a_ * b_)[m].sum() / (np.sqrt((a_[m] ** 2).sum() * (b_[m] ** 2).sum()) + 1e-9)
        ncc0[i] = c
        core = a_[8:-8, 8:-8]
        res = cv2.matchTemplate(b_, core, cv2.TM_CCOEFF_NORMED)
        k = np.unravel_index(np.argmax(res), res.shape)
        bestncc[i] = res[k]; shift[i] = (k[1] - 8, k[0] - 8)
        rr0, cc0 = r0 + y, c0 + x
        origin[i] = (rr0, cc0, t.f + (rr0 + P / 2) * t.e, t.c + (cc0 + P / 2) * s)
        split[i] = 1 if ((y // (4 * S)) % args.val_every == args.val_every - 1) else 0

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "patches.npz", ohrc=A, tmc=B, mask=M, split=split,
                        origin_row_col_lat_lon=origin, qa_ncc_at_zero_shift=ncc0,
                        qa_best_ncc_within_8px=bestncc, qa_best_shift_xy_px=shift)
    index = {
        "n_patches": N, "patch_px": P, "stride_px": S, "tmc_pixel_m": round(s * M_PER_DEG, 3),
        "n_train": int((split == 0).sum()), "n_val": int((split == 1).sum()),
        "registration_offset_m": {"east": args.offset_east_m, "north": args.offset_north_m},
        "alignment_status": ("nominal geometry, no offset" if args.offset_east_m == args.offset_north_m == 0
                             else "manual offset supplied by user; not verified by this script"),
        "qa_summary": {"ncc0_median": float(np.median(ncc0)), "best_ncc_median": float(np.median(bestncc)),
                       "median_best_shift_px": [float(np.median(shift[:, 0])), float(np.median(shift[:, 1]))],
                       "note": "cross-illumination NCC is low even when aligned; a consistent non-zero "
                               "median shift means the supplied offset is off by that many TMC pixels."},
        "sources": {"ohrc": ohrc_p.folder, "tmc": tmc_p.folder}, "tensor_format": "NCHW float32 [0,1]",
    }
    (args.out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    pick = np.linspace(0, N - 1, min(N, 8)).astype(int)
    strip = np.vstack([np.hstack([A[i, 0], B[i, 0]]) for i in pick])
    cv2.imwrite(str(args.out / "preview.png"), (strip * 255).astype(np.uint8))
    print(f"wrote {N} patches ({index['n_train']} train / {index['n_val']} val) -> {args.out}")
    print(json.dumps(index["qa_summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
