"""
MAGSAC++ evaluation of KAN-LoFTR on held-out patch pairs.

    python src/evaluate.py --data outputs/pairs/pairs.npz --checkpoint outputs/kan_loftr/checkpoint.pt --split val

For every pair two match sets are scored, so the KAN's contribution is visible:
    coarse : keypoints1 = coarse cell centre (no KAN offset)
    kan    : keypoints1 = coarse cell centre + KAN sub-pixel offset
Each set goes through cv2.findHomography(..., cv2.USAC_MAGSAC, 1.5) and reports
    inlier %, the estimated H,
    rmse_inliers  = sqrt(mean ||x' - H x||^2) over MAGSAC inliers (self-consistency),
    rmse_gt       = same against the ground-truth homography H_gt, over ALL matches,
    corner_err    = mean distance of the 4 patch corners under H vs H_gt (px).
Without --checkpoint the KAN head is untrained, so "kan" is meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import PairDataset  # noqa: E402
from model import KANLoFTR  # noqa: E402

OUTPUTS_DIR = Path("outputs")          # relative to the working directory; pass --data/--out explicitly on Kaggle

MAGSAC_THRESH_PX = 1.5
SUCCESS_CORNER_ERR_PX = 3.0


def project(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H.astype(np.float64)).reshape(-1, 2)


def score_matches(kp0: np.ndarray, kp1: np.ndarray, H_gt: np.ndarray, patch: int) -> dict:
    """MAGSAC++ homography and the error metrics for one pair (see module docstring)."""
    n = len(kp0)
    out = {"n_matches": n, "ok": False}
    if n < 4:
        return out
    H, mask = cv2.findHomography(kp0.astype(np.float32), kp1.astype(np.float32), cv2.USAC_MAGSAC,
                                 MAGSAC_THRESH_PX, maxIters=10000, confidence=0.999)
    if H is None or mask is None:
        return out
    inl = mask.ravel().astype(bool)
    corners = np.array([[0, 0], [patch - 1, 0], [patch - 1, patch - 1], [0, patch - 1]], np.float64)
    out.update(
        ok=True, H=H.tolist(), inlier_pct=100.0 * float(inl.mean()),
        rmse_inliers=float(np.sqrt(((kp1[inl] - project(kp0[inl], H)) ** 2).sum(-1).mean())),
        rmse_gt=float(np.sqrt(((kp1 - project(kp0, H_gt)) ** 2).sum(-1).mean())),
        corner_err=float(np.linalg.norm(project(corners, H) - project(corners, H_gt), axis=1).mean()),
    )
    return out


def summarise(results: list[dict]) -> dict:
    good = [r for r in results if r["ok"]]
    if not good:
        return {"pairs": len(results), "failed": len(results)}
    med = lambda k: float(np.median([r[k] for r in good]))  # noqa: E731
    return {
        "pairs": len(results), "failed": len(results) - len(good),
        "mean_matches": float(np.mean([r["n_matches"] for r in results])),
        "mean_inlier_pct": float(np.mean([r["inlier_pct"] for r in good])),
        "median_rmse_inliers_px": med("rmse_inliers"), "median_rmse_gt_px": med("rmse_gt"),
        "median_corner_err_px": med("corner_err"),
        "success_rate": float(np.mean([r["corner_err"] < SUCCESS_CORNER_ERR_PX for r in good]) * len(good) / len(results)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=OUTPUTS_DIR / "pairs" / "pairs.npz")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--pretrained", default="outdoor")
    ap.add_argument("--thr", type=float, default=None, help="coarse confidence threshold (default 0.2)")
    ap.add_argument("--max-perturb", type=float, default=16.0)
    ap.add_argument("--guard-px", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=5, help="different synthetic warps per patch")
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "eval")
    ap.add_argument("--verbose", action="store_true", help="print H per pair")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KANLoFTR(pretrained=None if args.pretrained.lower() == "none" else args.pretrained)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu")
        model.loftr_coarse.load_state_dict(ck["coarse"])
        model.kan.load_state_dict(ck["kan"])
        print(f"loaded {args.checkpoint} (epoch {ck.get('epoch')})")
    else:
        print("WARNING: no --checkpoint: KAN head is untrained, only the 'coarse' row is meaningful.")
    model.to(device).eval()

    ds = PairDataset(args.data, args.split, max_perturb=args.max_perturb, guard_px=args.guard_px,
                     photometric=0.0, deterministic=True, seed=4321, repeat=args.repeats)
    res = {"coarse": [], "kan": []}
    per_pair = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds.get_numpy(i)
            t = {k: torch.from_numpy(v)[None].to(device) for k, v in s.items() if k in ("image0", "image1", "mask0", "mask1")}
            o = model(t["image0"], t["image1"], t["mask0"], t["mask1"], thr=args.thr)
            kp0 = o["keypoints0"].cpu().numpy()
            for name, key in (("coarse", "keypoints1_coarse"), ("kan", "keypoints1")):
                r = score_matches(kp0, o[key].cpu().numpy(), s["H_gt"], ds.patch)
                res[name].append(r)
                if args.verbose and r["ok"]:
                    print(f"pair {i:3d} {name:6s} matches {r['n_matches']:4d} inliers {r['inlier_pct']:5.1f}% "
                          f"rmse {r['rmse_inliers']:.3f}px gt-rmse {r['rmse_gt']:.3f}px H={np.round(r['H'], 4).tolist()}")
            per_pair.append({"index": int(s["index"]), **{n: res[n][-1] for n in res}})

    summary = {n: summarise(res[n]) for n in res}
    print(f"\n{args.split} split: {len(ds)} evaluations ({len(ds.indices)} patches x {args.repeats} warps), "
          f"MAGSAC++ threshold {MAGSAC_THRESH_PX}px")
    keys = ["mean_matches", "mean_inlier_pct", "median_rmse_inliers_px", "median_rmse_gt_px",
            "median_corner_err_px", "success_rate", "failed"]
    print(f"{'':24s}{'coarse':>12s}{'kan':>12s}")
    for k in keys:
        print(f"{k:24s}" + "".join(f"{summary[n].get(k, float('nan')):12.3f}" for n in ("coarse", "kan")))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"metrics_{args.split}.json"
    path.write_text(json.dumps({"summary": summary, "pairs": per_pair, "args": {k: str(v) for k, v in vars(args).items()}},
                               indent=2), encoding="utf-8")
    print(f"telemetry -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
