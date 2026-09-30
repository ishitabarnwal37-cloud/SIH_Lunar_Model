"""
Mixed-precision fine-tuning of KAN-LoFTR.

    python src/train.py --data outputs/pairs/pairs.npz --epochs 30 --batch-size 8

Trainable: LoFTR coarse attention (lr 1e-4) and the KAN head (lr 1e-3, weight
decay 1e-2). The ResNet-FPN backbone stays frozen.

Loss = coarse_weight * coarse_matching_loss + kan_weight * MSE(KAN keypoint,
H_gt projection). The coarse matching loss is an addition to the plain KAN MSE:
the coarse match selection is not differentiable, so with only the offset MSE
the attention layers get no signal that keeps their matching accurate.
Use --coarse-weight 0 for the KAN-MSE-only objective.

Checkpoint (best validation KAN RMSE) stores only the trainable parts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import PairDataset  # noqa: E402
from model import KANLoFTR, coarse_matching_loss, homography_targets, kan_offset_loss  # noqa: E402

OUTPUTS_DIR = Path("outputs")          # relative to the working directory; pass --data/--out explicitly on Kaggle


def amp_tools(device: torch.device, enabled: bool):
    """(GradScaler, autocast context factory); works with old and new torch AMP APIs."""
    use = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use), lambda: torch.amp.autocast(device_type=device.type, enabled=use)
    return torch.cuda.amp.GradScaler(enabled=use), lambda: torch.cuda.amp.autocast(enabled=use)


def forward_losses(model, batch, device, ctx, args):
    im0, im1 = batch["image0"].to(device), batch["image1"].to(device)
    m0, m1 = batch["mask0"].to(device), batch["mask1"].to(device)
    H = batch["H_gt"].to(device)
    with ctx():
        enc = model.encode(im0, im1, m0, m1)
    j, ok, p1 = homography_targets(H, enc)
    l_coarse = coarse_matching_loss(enc, j, ok)
    l_kan, rmse, base = kan_offset_loss(model, enc, j, ok, p1, args.max_pairs)
    with torch.no_grad():
        top1 = (enc["logp"].argmax(2)[ok] == j[ok]).float().mean().item() if ok.any() else float("nan")
    loss = args.coarse_weight * l_coarse + args.kan_weight * l_kan
    return loss, {"loss": loss.item(), "coarse": l_coarse.item(), "kan_mse": l_kan.item(),
                  "kan_rmse_px": rmse, "no_offset_rmse_px": base, "coarse_top1": top1}


def mean_metrics(rows):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


@torch.no_grad()
def validate(model, loader, device, ctx, args):
    model.eval()
    rows = [forward_losses(model, b, device, ctx, args)[1] for b in loader]
    return mean_metrics(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=OUTPUTS_DIR / "pairs" / "pairs.npz")
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "kan_loftr")
    ap.add_argument("--pretrained", default="outdoor", help="kornia LoFTR weights ('outdoor', 'indoor') or 'none'")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=20, help="passes over the train patches per epoch (each with fresh warps)")
    ap.add_argument("--lr-coarse", type=float, default=1e-4)
    ap.add_argument("--lr-kan", type=float, default=1e-3)
    ap.add_argument("--wd-kan", type=float, default=1e-2)
    ap.add_argument("--wd-coarse", type=float, default=0.0)
    ap.add_argument("--coarse-weight", type=float, default=1.0)
    ap.add_argument("--kan-weight", type=float, default=1.0)
    ap.add_argument("--max-perturb", type=float, default=16.0)
    ap.add_argument("--guard-px", type=int, default=None, help="guard band between splits (default: one patch)")
    ap.add_argument("--max-pairs", type=int, default=1024, help="cell pairs per batch fed to the KAN")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = None if args.pretrained.lower() == "none" else args.pretrained

    train_ds = PairDataset(args.data, "train", max_perturb=args.max_perturb, guard_px=args.guard_px, repeat=args.repeat)
    val_ds = PairDataset(args.data, "val", max_perturb=args.max_perturb, guard_px=args.guard_px,
                         photometric=0.0, deterministic=True, seed=1234)
    print(f"device {device} | patch {train_ds.patch}px | splits {train_ds.split_counts} "
          f"(dropped by guard {train_ds.n_dropped_by_guard}) | train samples/epoch {len(train_ds)}")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          drop_last=True, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if len(train_dl) == 0:
        sys.exit("ERROR: fewer train samples than one batch; lower --batch-size or raise --repeat.")

    model = KANLoFTR(pretrained=pretrained).to(device)
    groups = model.param_groups()
    opt = torch.optim.AdamW([
        {"params": groups["coarse"], "lr": args.lr_coarse, "weight_decay": args.wd_coarse},
        {"params": groups["kan"], "lr": args.lr_kan, "weight_decay": args.wd_kan},
    ])
    n_train = sum(p.numel() for g in groups.values() for p in g)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train / 1e6:.2f}M of {n_all / 1e6:.2f}M parameters (coarse attention + KAN)")

    scaler, ctx = amp_tools(device, not args.no_amp)
    args.out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, rows, skipped = time.time(), [], 0
        for batch in train_dl:
            loss, stats = forward_losses(model, batch, device, ctx, args)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], args.clip)
            scaler.step(opt)
            scaler.update()
            rows.append(stats)
        tr = mean_metrics(rows) if rows else {}
        va = validate(model, val_dl, device, ctx, args)
        history.append({"epoch": epoch, "train": tr, "val": va, "skipped_nonfinite": skipped})
        print(f"epoch {epoch:3d} | train loss {tr.get('loss', float('nan')):.4f} kan_rmse {tr.get('kan_rmse_px', float('nan')):.3f}px"
              f" | val kan_rmse {va['kan_rmse_px']:.3f}px vs no-offset {va['no_offset_rmse_px']:.3f}px"
              f" top1 {va['coarse_top1']:.3f} | {time.time() - t0:.0f}s" + (f" | skipped {skipped}" if skipped else ""))
        if va["kan_rmse_px"] < best:
            best = va["kan_rmse_px"]
            torch.save({"coarse": model.loftr_coarse.state_dict(), "kan": model.kan.state_dict(),
                        "args": {k: str(v) for k, v in vars(args).items()}, "epoch": epoch, "val": va},
                       args.out / "checkpoint.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"best val KAN RMSE {best:.3f}px -> {args.out / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
