from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="runs")
    p.add_argument("--sampling-rate", type=int, choices=[100, 500], default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--monitor", type=str, choices=["auroc", "auprc", "loss"], default="auroc")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-dataset", action="store_true")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--input-clip", type=float, default=8.0)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--dilations", type=str, default="1,2,4,8,16,32,64")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--observer-momentum", type=float, default=0.95)
    p.add_argument("--pow2-scales", action="store_true")
    p.add_argument("--freeze-quant-epoch", type=int, default=-1)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--save-every", type=int, default=1)

    return p.parse_args()
