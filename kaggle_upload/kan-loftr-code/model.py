"""
KAN-LoFTR: Kornia's pretrained LoFTR with the fine stage replaced by a KAN.

    image0, image1 (B, 1, H, W) in [0, 1], H and W divisible by 8
      -> frozen ResNet-FPN backbone           (1/8 coarse features, 256-d)
      -> sine position encoding + trainable coarse self/cross attention (loftr_coarse)
      -> dual-softmax coarse matching         (conf = softmax_i * softmax_j)
      -> KAN([512, 64, 2]) on concat(f0[i], f1[j]) -> sub-cell offset (dx, dy) px
      -> refined keypoint in image1 = coarse cell centre + offset

Kornia's own fine stage (fine_preprocess, loftr_fine, fine_matching) is not
used. Kornia's LoFTR has no MLP fine head: `loftr_fine` is a fine-level
attention transformer, so the whole fine stage is what gets replaced.

Coordinates are pixel (x, y) with integer coordinates at pixel centres (the
OpenCV convention); coarse cell (u, v) is centred at ((u+.5)*8-.5, (v+.5)*8-.5).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kan import KAN

STRIDE = 8


def load_kornia_loftr(pretrained: str | None):
    try:
        from kornia.feature import LoFTR
    except ImportError as e:
        raise ImportError("kornia is required: pip install kornia einops") from e
    return LoFTR(pretrained=pretrained)          # pretrained=None -> random weights, no download


def cell_centers(hw, stride: int = STRIDE, device=None) -> torch.Tensor:
    """(h*w, 2) pixel (x, y) centres of the coarse cells, row-major like flatten(2)."""
    h, w = hw
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack([(xs.reshape(-1) + 0.5) * stride - 0.5, (ys.reshape(-1) + 0.5) * stride - 0.5], -1).float()


class KANLoFTR(nn.Module):
    def __init__(self, pretrained: str | None = "outdoor", kan_hidden: int = 64, grid_size: int = 5,
                 spline_order: int = 3, dsmax_temperature: float = 0.1, match_thr: float = 0.2,
                 border_rm: int = 2, max_offset: float = float(STRIDE)):
        super().__init__()
        loftr = load_kornia_loftr(pretrained)
        for name in ("backbone", "pos_encoding", "loftr_coarse"):
            if not hasattr(loftr, name):
                raise AttributeError(f"this kornia LoFTR has no `{name}`; kornia version is incompatible")
        self.backbone, self.pos_encoding, self.loftr_coarse = loftr.backbone, loftr.pos_encoding, loftr.loftr_coarse
        del loftr                                    # drops coarse_matching / fine_preprocess / loftr_fine / fine_matching

        for p in self.backbone.parameters():         # frozen ResNet-FPN
            p.requires_grad = False
        for p in self.loftr_coarse.parameters():     # trainable coarse self/cross attention
            p.requires_grad = True

        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(torch.zeros(2, 1, 64, 64))
        self.coarse_dim = int(feats[0].shape[1])
        if feats[0].shape[-2:] != (64 // STRIDE, 64 // STRIDE):
            raise RuntimeError(f"backbone coarse map is {tuple(feats[0].shape[-2:])}, expected stride {STRIDE}")

        self.kan = KAN([2 * self.coarse_dim, kan_hidden, 2], grid_size=grid_size, spline_order=spline_order)
        self.temperature, self.match_thr = dsmax_temperature, match_thr
        self.border_rm, self.max_offset = border_rm, max_offset

    # keep the frozen backbone's BatchNorm statistics fixed
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def param_groups(self) -> dict:
        return {"coarse": [p for p in self.loftr_coarse.parameters() if p.requires_grad],
                "kan": list(self.kan.parameters())}

    # ------------------------------------------------------------------ encode
    def encode(self, image0: torch.Tensor, image1: torch.Tensor,
               mask0: torch.Tensor | None = None, mask1: torch.Tensor | None = None) -> dict:
        if image0.dim() != 4 or image0.shape[1] != 1 or image0.shape != image1.shape:
            raise ValueError(f"need two (B, 1, H, W) images of equal shape, got {tuple(image0.shape)} / {tuple(image1.shape)}")
        b, _, h, w = image0.shape
        if h % STRIDE or w % STRIDE:
            raise ValueError(f"image size {h}x{w} must be divisible by {STRIDE}")
        hc, wc = h // STRIDE, w // STRIDE

        with torch.no_grad():                        # backbone is frozen
            feats_c = self.backbone(torch.cat([image0, image1], 0))[0]
        if feats_c.shape != (2 * b, self.coarse_dim, hc, wc):
            raise RuntimeError(f"backbone output {tuple(feats_c.shape)} != {(2 * b, self.coarse_dim, hc, wc)}")
        c0, c1 = feats_c.split(b)

        f0 = self.pos_encoding(c0).flatten(2).transpose(1, 2)      # (B, L, C)
        f1 = self.pos_encoding(c1).flatten(2).transpose(1, 2)
        f0, f1 = self.loftr_coarse(f0, f1)
        L = hc * wc
        if f0.shape != (b, L, self.coarse_dim) or f1.shape != (b, L, self.coarse_dim):
            raise RuntimeError(f"coarse transformer output {tuple(f0.shape)} != {(b, L, self.coarse_dim)}")

        valid0 = self._coarse_valid(mask0, b, hc, wc, image0.device)
        valid1 = self._coarse_valid(mask1, b, hc, wc, image0.device)
        with torch.autocast(device_type=image0.device.type, enabled=False):      # matching stays in fp32
            n0, n1 = f0.float() / math.sqrt(self.coarse_dim), f1.float() / math.sqrt(self.coarse_dim)
            sim = torch.einsum("blc,bsc->bls", n0, n1) / self.temperature
            sim = sim.masked_fill(~(valid0[:, :, None] & valid1[:, None, :]), -1e4)
            logp = F.log_softmax(sim, dim=2) + F.log_softmax(sim, dim=1)     # log of the dual-softmax confidence
        return {"feat0": f0, "feat1": f1, "logp": logp, "hw": (hc, wc), "valid0": valid0, "valid1": valid1}

    @staticmethod
    def _coarse_valid(mask, b, hc, wc, device) -> torch.Tensor:
        if mask is None:
            return torch.ones(b, hc * wc, dtype=torch.bool, device=device)
        return (F.avg_pool2d(mask.float(), STRIDE) > 0.5).flatten(1)

    # ------------------------------------------------------------------- match
    @torch.no_grad()
    def match(self, enc: dict, thr: float | None = None) -> dict:
        """Mutual-nearest-neighbour matches above `thr`, away from the border."""
        thr = self.match_thr if thr is None else thr
        conf = enc["logp"].exp()
        b, L0, L1 = conf.shape
        hc, wc = enc["hw"]
        row_max, row_arg = conf.max(2)
        _, col_arg = conf.max(1)
        mutual = col_arg.gather(1, row_arg) == torch.arange(L0, device=conf.device)[None]
        ys, xs = torch.meshgrid(torch.arange(hc, device=conf.device), torch.arange(wc, device=conf.device), indexing="ij")
        inner = ((xs >= self.border_rm) & (xs < wc - self.border_rm) & (ys >= self.border_rm) & (ys < hc - self.border_rm)).reshape(-1)
        keep = mutual & (row_max > thr) & enc["valid0"] & inner[None] & enc["valid1"].gather(1, row_arg)
        bi, ii = keep.nonzero(as_tuple=True)
        return {"b": bi, "i": ii, "j": row_arg[bi, ii], "conf": row_max[bi, ii]}

    # ------------------------------------------------------------------ refine
    def refine(self, enc: dict, b: torch.Tensor, i: torch.Tensor, j: torch.Tensor):
        """KAN sub-cell offsets for matched cells. Returns (kp0, kp1_coarse, kp1_refined), each (M, 2)."""
        x = torch.cat([enc["feat0"][b, i], enc["feat1"][b, j]], dim=-1)
        if x.shape[-1] != self.kan.in_features:
            raise RuntimeError(f"KAN input dim {x.shape[-1]} != {self.kan.in_features}")
        raw = self.kan(x)
        if raw.shape != (x.shape[0], 2):
            raise RuntimeError(f"KAN output {tuple(raw.shape)} != {(x.shape[0], 2)}")
        offset = self.max_offset * torch.tanh(raw / self.max_offset)      # smooth bound at +-max_offset px
        centers = cell_centers(enc["hw"], device=x.device)
        kp0, kp1c = centers[i], centers[j]
        return kp0, kp1c, kp1c + offset

    # ----------------------------------------------------------------- forward
    def forward(self, image0, image1, mask0=None, mask1=None, thr: float | None = None) -> dict:
        enc = self.encode(image0, image1, mask0, mask1)
        m = self.match(enc, thr)
        kp0, kp1c, kp1 = self.refine(enc, m["b"], m["i"], m["j"])
        return {"keypoints0": kp0, "keypoints1_coarse": kp1c, "keypoints1": kp1,
                "confidence": m["conf"], "batch_indexes": m["b"]}


# ------------------------------------------------------------------ supervision
def homography_targets(H: torch.Tensor, enc: dict, stride: int = STRIDE):
    """
    Ground truth from H_gt (image0 px -> image1 px, (B, 3, 3)).
    Returns j_gt (B, L) long: the image1 coarse cell each image0 cell lands in,
            ok   (B, L) bool: that landing is inside image1 and both cells are valid,
            p1   (B, L, 2):  the exact image1 pixel position of every image0 cell centre.
    """
    b = H.shape[0]
    hc, wc = enc["hw"]
    c = cell_centers((hc, wc), stride, H.device)
    ph = torch.cat([c, torch.ones_like(c[:, :1])], -1)                       # (L, 3)
    p = torch.einsum("bij,lj->bli", H.float(), ph)                           # (B, L, 3)
    z = p[..., 2:3]
    z = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
    p1 = p[..., :2] / z
    u = torch.round((p1[..., 0] + 0.5) / stride - 0.5).long()
    v = torch.round((p1[..., 1] + 0.5) / stride - 0.5).long()
    inside = (u >= 0) & (u < wc) & (v >= 0) & (v < hc)
    j = v.clamp(0, hc - 1) * wc + u.clamp(0, wc - 1)
    ok = inside & enc["valid0"] & enc["valid1"].gather(1, j)
    return j, ok, p1


def coarse_matching_loss(enc: dict, j_gt: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """-log dual-softmax confidence at the ground-truth cell pairs (mean over valid cells)."""
    if not ok.any():
        return enc["logp"].sum() * 0.0
    bi, ii = ok.nonzero(as_tuple=True)
    return -enc["logp"][bi, ii, j_gt[bi, ii]].mean()


def kan_offset_loss(model: KANLoFTR, enc: dict, j_gt, ok, p1, max_pairs: int = 1024):
    """
    MSE between KAN-refined keypoints and the H_gt projection of the image0 cell centres,
    on ground-truth cell pairs (teacher forcing). Returns (loss, rmse_px, baseline_rmse_px);
    the baseline is the coarse cell centre without any offset.
    """
    bi, ii = ok.nonzero(as_tuple=True)
    if len(bi) == 0:
        z = enc["logp"].sum() * 0.0
        return z, float("nan"), float("nan")
    if len(bi) > max_pairs:
        pick = torch.randperm(len(bi), device=bi.device)[:max_pairs]
        bi, ii = bi[pick], ii[pick]
    jj = j_gt[bi, ii]
    _, kp1c, kp1 = model.refine(enc, bi, ii, jj)
    target = p1[bi, ii]
    loss = F.mse_loss(kp1, target)
    with torch.no_grad():
        rmse = ((kp1 - target) ** 2).sum(-1).mean().sqrt().item()
        base = ((kp1c - target) ** 2).sum(-1).mean().sqrt().item()
    return loss, rmse, base
