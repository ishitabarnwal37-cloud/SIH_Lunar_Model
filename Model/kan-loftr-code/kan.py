"""
B-spline Kolmogorov-Arnold layers.

Self-contained re-implementation of the efficient-kan algorithm
(Blealtan/efficient-kan, MIT licence): each edge carries a learnable
function  phi(x) = w_b * SiLU(x) + w_s * sum_k c_k B_k(x),  evaluated as a
dense B-spline basis expansion so it runs as two matmuls.

    KAN([512, 64, 2], grid_size=5, spline_order=3)

Splines are always evaluated in fp32 (autocast disabled) because the
Cox-de Boor recursion divides by knot differences.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_range: Sequence[float] = (-1.0, 1.0),
    ):
        super().__init__()
        if grid_size < 1 or spline_order < 1:
            raise ValueError("grid_size and spline_order must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        h = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())   # (in, G + 2k + 1)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) \
                * self.scale_noise / self.grid_size
            self.spline_weight.copy_(self._curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """B-spline bases of order `spline_order` at x: (N, in) -> (N, in, grid_size + spline_order)."""
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"b_splines expects (N, {self.in_features}), got {tuple(x.shape)}")
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
                + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Least-squares spline coefficients that interpolate y at x: -> (out, in, G + k)."""
        a = self.b_splines(x).transpose(0, 1)                    # (in, N, G + k)
        b = y.transpose(0, 1)                                    # (in, N, out)
        return torch.linalg.lstsq(a, b).solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(f"KANLinear expects last dim {self.in_features}, got {tuple(x.shape)}")
        lead = x.shape[:-1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.reshape(-1, self.in_features).float()
            base = F.linear(self.base_activation(x), self.base_weight)
            spline = F.linear(self.b_splines(x).reshape(x.size(0), -1),
                              self.scaled_spline_weight.reshape(self.out_features, -1))
            out = base + spline
        return out.reshape(*lead, self.out_features)


class KAN(nn.Module):
    """Stack of KANLinear layers with a parameter-free LayerNorm in front of each,
    so activations land inside the spline grid range."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation=nn.SiLU,
        grid_range: Sequence[float] = (-3.0, 3.0),
    ):
        super().__init__()
        if len(layers_hidden) < 2:
            raise ValueError("layers_hidden needs at least an input and an output size")
        self.in_features = layers_hidden[0]
        self.out_features = layers_hidden[-1]
        self.norms = nn.ModuleList(nn.LayerNorm(d, elementwise_affine=False) for d in layers_hidden[:-1])
        self.layers = nn.ModuleList(
            KANLinear(i, o, grid_size, spline_order, base_activation=base_activation, grid_range=grid_range)
            for i, o in zip(layers_hidden[:-1], layers_hidden[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, layer in zip(self.norms, self.layers):
            x = layer(norm(x.float()))
        return x
