from __future__ import annotations

from typing import Tuple

import torch

from .constants import INT32_QMAX


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    return x + (x.clamp(qmin, qmax) - x).detach()


def fake_quant_sym(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.clamp(scale, min=1e-8)
    x_int = clamp_ste(round_ste(x / scale), qmin, qmax)
    x_deq = x_int * scale
    return x_deq, x_int


def fake_quant_sym_per_channel(
    x: torch.Tensor,
    scale: torch.Tensor,
    ch_axis: int,
    qmin: int,
    qmax: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = [1] * x.ndim
    shape[ch_axis] = -1
    scale_view = torch.clamp(scale.view(*shape), min=1e-8)
    x_int = clamp_ste(round_ste(x / scale_view), qmin, qmax)
    x_deq = x_int * scale_view
    return x_deq, x_int


def integer_requantize(
    acc: torch.Tensor,
    multiplier: torch.Tensor,
    shift: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    view_shape = [1] * acc.ndim
    view_shape[1] = -1
    m = multiplier.view(*view_shape).float()
    y = acc * (m / float(1 << shift))
    y_int = clamp_ste(round_ste(y), qmin, qmax)
    return y_int


def approx_int_multiplier(real_multiplier: torch.Tensor, shift: int = 24) -> torch.Tensor:
    real_multiplier = torch.clamp(real_multiplier, min=1e-12, max=128.0)
    mult = torch.round(real_multiplier * float(1 << shift)).clamp(1, INT32_QMAX)
    return mult.to(torch.int64)
