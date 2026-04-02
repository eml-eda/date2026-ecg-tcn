from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    INT16_QMAX,
    INT16_QMIN,
    INT32_QMAX,
    INT32_QMIN,
    INT8_QMAX,
    INT8_QMIN,
    LABELS,
)
from .quantization import (
    approx_int_multiplier,
    clamp_ste,
    fake_quant_sym,
    fake_quant_sym_per_channel,
    integer_requantize,
    round_ste,
)


class EMASymmetricObserver(nn.Module):
    def __init__(
        self,
        num_bits: int,
        signed: bool,
        momentum: float = 0.95,
        eps: float = 1e-8,
        per_channel: bool = False,
        ch_axis: int = 0,
        num_channels: Optional[int] = None,
        pow2_scale: bool = False,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.signed = signed
        self.momentum = momentum
        self.eps = eps
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.pow2_scale = pow2_scale

        if per_channel:
            if num_channels is None:
                raise ValueError("num_channels must be provided for per_channel observer")
            init_scale = torch.ones(num_channels, dtype=torch.float32)
        else:
            init_scale = torch.tensor(1.0, dtype=torch.float32)

        self.register_buffer("scale", init_scale)
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("frozen", torch.tensor(False))

    @property
    def qmax(self) -> int:
        if self.signed:
            return 2 ** (self.num_bits - 1) - 1
        return 2**self.num_bits - 1

    def _snap_scale(self, scale: torch.Tensor) -> torch.Tensor:
        if not self.pow2_scale:
            return scale
        return torch.pow(2.0, torch.round(torch.log2(torch.clamp(scale, min=self.eps))))

    def freeze(self) -> None:
        self.frozen.fill_(True)

    def unfreeze(self) -> None:
        self.frozen.fill_(False)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if bool(self.frozen.item()):
            return

        x_det = x.detach().float()
        if self.per_channel:
            dims = [i for i in range(x_det.ndim) if i != self.ch_axis]
            amax = x_det.abs().amax(dim=dims)
        else:
            amax = x_det.abs().max()

        new_scale = torch.clamp(amax / float(self.qmax), min=self.eps)
        new_scale = self._snap_scale(new_scale)

        if not bool(self.initialized.item()):
            self.scale.copy_(new_scale)
            self.initialized.fill_(True)
        else:
            self.scale.mul_(self.momentum).add_(new_scale * (1.0 - self.momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.update(x)
        return self.scale


class IntegerModule(nn.Module):
    def freeze_observers(self) -> None:
        for m in self.modules():
            if isinstance(m, EMASymmetricObserver):
                m.freeze()

    def unfreeze_observers(self) -> None:
        for m in self.modules():
            if isinstance(m, EMASymmetricObserver):
                m.unfreeze()


class IntegerConv1dQAT(IntegerModule):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = True,
        causal: bool = False,
        observer_momentum: float = 0.95,
        pow2_scales: bool = False,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.causal = causal

        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self.act_in_obs = EMASymmetricObserver(
            num_bits=8,
            signed=True,
            momentum=observer_momentum,
            per_channel=False,
            pow2_scale=pow2_scales,
        )
        self.weight_obs = EMASymmetricObserver(
            num_bits=8,
            signed=True,
            momentum=observer_momentum,
            per_channel=True,
            ch_axis=0,
            num_channels=out_ch,
            pow2_scale=pow2_scales,
        )
        self.act_out_obs = EMASymmetricObserver(
            num_bits=16,
            signed=True,
            momentum=observer_momentum,
            per_channel=False,
            pow2_scale=pow2_scales,
        )
        self.requant_shift = 24

    @property
    def context_len(self) -> int:
        return (self.kernel_size - 1) * self.dilation

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        total = self.context_len
        if total <= 0:
            return x
        if self.causal:
            return F.pad(x, (total, 0))
        left = total // 2
        right = total - left
        return F.pad(x, (left, right))

    def _run_quant_conv(self, x_aligned: torch.Tensor) -> torch.Tensor:
        x_scale = self.act_in_obs.scale.detach()
        w_scale = self.weight_obs.scale.detach()

        _, x_int = fake_quant_sym(x_aligned, x_scale, INT8_QMIN, INT8_QMAX)
        _, w_int = fake_quant_sym_per_channel(
            self.weight,
            w_scale,
            ch_axis=0,
            qmin=INT8_QMIN,
            qmax=INT8_QMAX,
        )

        acc = F.conv1d(x_int, w_int, bias=None, stride=self.stride, dilation=self.dilation)
        acc_scale = (x_scale * w_scale).view(1, -1, 1)

        if self.bias is not None:
            b_int = clamp_ste(round_ste(self.bias.view(1, -1, 1) / acc_scale), INT32_QMIN, INT32_QMAX)
            acc = acc + b_int

        float_out = acc * acc_scale
        if self.training:
            self.act_out_obs.update(float_out)

        y_scale = self.act_out_obs.scale.detach()
        ratio = (x_scale * w_scale) / y_scale
        multiplier = approx_int_multiplier(ratio, shift=self.requant_shift)
        y_int = integer_requantize(acc, multiplier, self.requant_shift, INT16_QMIN, INT16_QMAX)
        return y_int * y_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.act_in_obs.update(x)
            self.weight_obs.update(self.weight)
        return self._run_quant_conv(self._pad(x))

    @torch.no_grad()
    def make_streaming_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.in_ch, self.context_len, device=device, dtype=dtype)

    def forward_step(self, x_t: torch.Tensor, state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.stride != 1:
            raise NotImplementedError("Streaming step is implemented only for stride=1")
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(-1)
        if x_t.ndim != 3 or x_t.shape[-1] != 1:
            raise ValueError("x_t must have shape [B, C] or [B, C, 1]")
        if not self.causal:
            raise RuntimeError("forward_step requires causal convolutions")

        if self.training:
            self.act_in_obs.update(x_t)
            self.weight_obs.update(self.weight)

        if state is None:
            state = self.make_streaming_state(x_t.shape[0], x_t.device, x_t.dtype)

        if self.context_len > 0:
            full = torch.cat([state, x_t], dim=-1)
            next_state = full[:, :, -self.context_len :].detach()
        else:
            full = x_t
            next_state = state

        y = self._run_quant_conv(full)
        return y.squeeze(-1), next_state

    @torch.no_grad()
    def export_int_state(self, name: str) -> Dict:
        x_scale = float(self.act_in_obs.scale.detach().cpu().item())
        w_scale = self.weight_obs.scale.detach().cpu().numpy().astype(np.float32)
        y_scale = float(self.act_out_obs.scale.detach().cpu().item())
        ratio = (self.act_in_obs.scale.detach() * self.weight_obs.scale.detach()) / self.act_out_obs.scale.detach()
        multiplier = approx_int_multiplier(ratio, shift=self.requant_shift).cpu().numpy().astype(np.int64)

        w_int = torch.round(
            self.weight.detach().cpu() / self.weight_obs.scale.detach().cpu().view(-1, 1, 1)
        ).clamp(INT8_QMIN, INT8_QMAX).to(torch.int8)

        b_int = None
        if self.bias is not None:
            bias_scale = self.act_in_obs.scale.detach().cpu() * self.weight_obs.scale.detach().cpu()
            b_int = torch.round(self.bias.detach().cpu() / bias_scale).clamp(INT32_QMIN, INT32_QMAX).to(torch.int32)

        return {
            "name": name,
            "type": "IntegerConv1dQAT",
            "in_channels": self.in_ch,
            "out_channels": self.out_ch,
            "kernel_size": self.kernel_size,
            "dilation": self.dilation,
            "stride": self.stride,
            "causal": self.causal,
            "streaming_context_len": self.context_len,
            "x_scale": x_scale,
            "w_scale_per_out_channel": w_scale.tolist(),
            "y_scale": y_scale,
            "requant_multiplier_per_out_channel": multiplier.tolist(),
            "requant_shift": self.requant_shift,
            "weight_int8": w_int.numpy(),
            "bias_int32": None if b_int is None else b_int.numpy(),
        }


class IntegerReLUQAT(IntegerModule):
    def __init__(self, observer_momentum: float = 0.95, pow2_scales: bool = False) -> None:
        super().__init__()
        self.obs = EMASymmetricObserver(
            num_bits=16,
            signed=True,
            momentum=observer_momentum,
            per_channel=False,
            pow2_scale=pow2_scales,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_relu = F.relu(x)
        if self.training:
            self.obs.update(x_relu)
        scale = self.obs.scale.detach()
        _, x_int = fake_quant_sym(x_relu, scale, 0, INT16_QMAX)
        return x_int * scale

    @torch.no_grad()
    def export_int_state(self, name: str) -> Dict:
        return {
            "name": name,
            "type": "IntegerReLUQAT",
            "scale": float(self.obs.scale.detach().cpu().item()),
        }


class IntegerAddReLUQAT(IntegerModule):
    def __init__(self, observer_momentum: float = 0.95, pow2_scales: bool = False, relu: bool = True) -> None:
        super().__init__()
        self.relu = relu
        self.obs = EMASymmetricObserver(
            num_bits=16,
            signed=True,
            momentum=observer_momentum,
            per_channel=False,
            pow2_scale=pow2_scales,
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        s = a + b
        if self.relu:
            s = F.relu(s)
        if self.training:
            self.obs.update(s)
        scale = self.obs.scale.detach()
        _, a_int = fake_quant_sym(a, scale, INT16_QMIN, INT16_QMAX)
        _, b_int = fake_quant_sym(b, scale, INT16_QMIN, INT16_QMAX)
        y_int = clamp_ste(a_int + b_int, INT16_QMIN, INT16_QMAX)
        if self.relu:
            y_int = clamp_ste(y_int, 0, INT16_QMAX)
        return y_int * scale

    @torch.no_grad()
    def export_int_state(self, name: str) -> Dict:
        return {
            "name": name,
            "type": "IntegerAddReLUQAT",
            "relu": self.relu,
            "scale": float(self.obs.scale.detach().cpu().item()),
        }


class IntegerGlobalAvgPool1dQAT(IntegerModule):
    def __init__(self, observer_momentum: float = 0.95, pow2_scales: bool = False) -> None:
        super().__init__()
        self.obs = EMASymmetricObserver(
            num_bits=16,
            signed=True,
            momentum=observer_momentum,
            per_channel=False,
            pow2_scale=pow2_scales,
        )
        self.recip_shift = 24

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.obs.update(x)
        scale = self.obs.scale.detach()
        _, x_int = fake_quant_sym(x, scale, INT16_QMIN, INT16_QMAX)
        acc = x_int.sum(dim=-1)
        length = x.shape[-1]
        recip = int(round((1 << self.recip_shift) / float(length)))
        y = acc * (float(recip) / float(1 << self.recip_shift))
        y_int = clamp_ste(round_ste(y), INT16_QMIN, INT16_QMAX)
        return y_int * scale

    @torch.no_grad()
    def export_int_state(self, name: str) -> Dict:
        return {
            "name": name,
            "type": "IntegerGlobalAvgPool1dQAT",
            "scale": float(self.obs.scale.detach().cpu().item()),
            "reciprocal_shift": self.recip_shift,
        }


class TCNBlock(IntegerModule):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool,
        observer_momentum: float,
        pow2_scales: bool,
    ) -> None:
        super().__init__()
        self.conv1 = IntegerConv1dQAT(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            bias=True,
            causal=causal,
            observer_momentum=observer_momentum,
            pow2_scales=pow2_scales,
        )
        self.relu1 = IntegerReLUQAT(observer_momentum=observer_momentum, pow2_scales=pow2_scales)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = IntegerConv1dQAT(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            bias=True,
            causal=causal,
            observer_momentum=observer_momentum,
            pow2_scales=pow2_scales,
        )
        self.drop2 = nn.Dropout(dropout)
        self.add_relu = IntegerAddReLUQAT(
            observer_momentum=observer_momentum,
            pow2_scales=pow2_scales,
            relu=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.drop1(y)
        y = self.conv2(y)
        y = self.drop2(y)
        y = self.add_relu(y, residual)
        return y

    def make_streaming_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            "conv1": self.conv1.make_streaming_state(batch_size, device, dtype),
            "conv2": self.conv2.make_streaming_state(batch_size, device, dtype),
        }

    def forward_step(self, x_t: torch.Tensor, state: Optional[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if state is None:
            state = self.make_streaming_state(x_t.shape[0], x_t.device, x_t.dtype)
        residual = x_t
        y, state1 = self.conv1.forward_step(x_t, state.get("conv1"))
        y = self.relu1(y)
        y, state2 = self.conv2.forward_step(y, state.get("conv2"))
        y = self.add_relu(y, residual)
        return y, {"conv1": state1, "conv2": state2}


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


class IntegerTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
        causal: bool,
        observer_momentum: float,
        pow2_scales: bool,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = list(dilations)
        self.causal = True if not causal else causal

        self.stem = IntegerConv1dQAT(
            in_channels,
            channels,
            kernel_size=kernel_size,
            dilation=1,
            stride=1,
            bias=True,
            causal=True,
            observer_momentum=observer_momentum,
            pow2_scales=pow2_scales,
        )
        self.stem_relu = IntegerReLUQAT(observer_momentum=observer_momentum, pow2_scales=pow2_scales)
        self.blocks = nn.ModuleList(
            [
                TCNBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                    causal=True,
                    observer_momentum=observer_momentum,
                    pow2_scales=pow2_scales,
                )
                for d in dilations
            ]
        )
        self.head = IntegerConv1dQAT(
            channels,
            num_classes,
            kernel_size=1,
            dilation=1,
            stride=1,
            bias=True,
            causal=True,
            observer_momentum=observer_momentum,
            pow2_scales=pow2_scales,
        )

    def receptive_field(self) -> int:
        rf = 1
        rf += self.kernel_size - 1
        for d in self.dilations:
            rf += 2 * (self.kernel_size - 1) * d
        return rf

    def streaming_state_spec(self) -> List[Dict[str, int]]:
        spec = [
            {
                "name": "stem",
                "channels": self.stem.in_ch,
                "context_len": self.stem.context_len,
            }
        ]
        for i, block in enumerate(self.blocks):
            spec.append(
                {
                    "name": f"blocks.{i}.conv1",
                    "channels": block.conv1.in_ch,
                    "context_len": block.conv1.context_len,
                }
            )
            spec.append(
                {
                    "name": f"blocks.{i}.conv2",
                    "channels": block.conv2.in_ch,
                    "context_len": block.conv2.context_len,
                }
            )
        spec.append(
            {
                "name": "head",
                "channels": self.head.in_ch,
                "context_len": self.head.context_len,
            }
        )
        return spec

    def make_streaming_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, object]:
        return {
            "stem": self.stem.make_streaming_state(batch_size, device, dtype),
            "blocks": [block.make_streaming_state(batch_size, device, dtype) for block in self.blocks],
            "head": self.head.make_streaming_state(batch_size, device, dtype),
        }

    def forward_step(self, x_t: torch.Tensor, state: Optional[Dict[str, object]] = None) -> Tuple[torch.Tensor, Dict[str, object]]:
        if state is None:
            state = self.make_streaming_state(x_t.shape[0], x_t.device, x_t.dtype)
        y, stem_state = self.stem.forward_step(x_t, state["stem"])
        y = self.stem_relu(y)
        new_block_states = []
        for block, block_state in zip(self.blocks, state["blocks"]):
            y, next_block_state = block.forward_step(y, block_state)
            new_block_states.append(next_block_state)
        logits, head_state = self.head.forward_step(y, state["head"])
        return logits, {"stem": stem_state, "blocks": new_block_states, "head": head_state}

    @torch.no_grad()
    def forward_streaming_reference(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        state = self.make_streaming_state(x.shape[0], x.device, x.dtype)
        logits = None
        for t in range(x.shape[-1]):
            logits, state = self.forward_step(x[:, :, t], state)
        if was_training:
            self.train()
        assert logits is not None
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x[:, :, -1]

    def freeze_observers(self) -> None:
        for m in self.modules():
            if isinstance(m, EMASymmetricObserver):
                m.freeze()

    def export_integer_state(self) -> Dict:
        payload = {
            "format": "ptbxl_integer_tcn_qat_v2_streaming",
            "labels": LABELS,
            "streaming": {
                "causal_only": True,
                "output_mode": "last_timestep",
                "receptive_field_samples": self.receptive_field(),
                "warmup_context_samples": self.receptive_field() - 1,
                "state_spec": self.streaming_state_spec(),
            },
            "layers": [],
        }
        for name, module in self.named_modules():
            if name == "":
                continue
            if hasattr(module, "export_int_state"):
                payload["layers"].append(module.export_int_state(name))
        return payload
