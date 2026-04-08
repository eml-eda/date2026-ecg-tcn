from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((pad, 0), 0.0)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class PrunableTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.drop2 = nn.Dropout(dropout)
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = x + residual
        return self.out_relu(x)


class PrunableTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: int,
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = list(dilations)

        self.stem = CausalConv1d(in_channels, channels, kernel_size=kernel_size, dilation=1)
        self.stem_relu = nn.ReLU()
        self.blocks = nn.ModuleList(
            [
                PrunableTCNBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in self.dilations
            ]
        )
        self.head = CausalConv1d(channels, num_classes, kernel_size=1, dilation=1)

    def receptive_field(self) -> int:
        rf = 1
        rf += self.kernel_size - 1
        for dilation in self.dilations:
            rf += 2 * (self.kernel_size - 1) * dilation
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x[:, :, -1]


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
