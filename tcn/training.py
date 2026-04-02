from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cli import parse_args
from .constants import LABELS
from .data import build_loaders, build_splits, load_ptbxl
from .model import IntegerTCN, unwrap_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(labels: np.ndarray, train_idx: np.ndarray) -> torch.Tensor:
    y = labels[train_idx]
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = (neg + 1e-6) / (pos + 1e-6)
    return torch.tensor(pos_weight, dtype=torch.float32)


def safe_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int32)

    aurocs = []
    auprcs = []
    f1s = []
    out = {}

    for i, label in enumerate(LABELS):
        if len(np.unique(y_true[:, i])) >= 2:
            auroc_i = roc_auc_score(y_true[:, i], y_prob[:, i])
            aurocs.append(auroc_i)
            out[f"auroc_{label}"] = float(auroc_i)
        else:
            out[f"auroc_{label}"] = float("nan")

        if y_true[:, i].sum() > 0:
            auprc_i = average_precision_score(y_true[:, i], y_prob[:, i])
            auprcs.append(auprc_i)
            out[f"auprc_{label}"] = float(auprc_i)
        else:
            out[f"auprc_{label}"] = float("nan")

        f1_i = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1s.append(f1_i)
        out[f"f1_{label}"] = float(f1_i)

    out["macro_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    out["macro_auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")
    out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses = []
    probs_all = []
    targets_all = []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(float(loss.item()))
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
        targets_all.append(y.cpu().numpy())

    y_true = np.concatenate(targets_all, axis=0)
    y_prob = np.concatenate(probs_all, axis=0)
    metrics = safe_classification_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    losses = []
    probs_all = []
    targets_all = []

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        probs_all.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

    y_true = np.concatenate(targets_all, axis=0)
    y_prob = np.concatenate(probs_all, axis=0)
    metrics = safe_classification_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs) / float(max(epochs - warmup_epochs, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_history_csv(history: List[Dict], path: Path) -> None:
    if not history:
        return
    keys = sorted({k for row in history for k in row.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow(row)


def save_plots(history: List[Dict], outdir: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    def plot_metric(metric_train: str, metric_val: str, metric_test: str, title: str, filename: str):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row.get(metric_train, np.nan) for row in history], label="train")
        plt.plot(epochs, [row.get(metric_val, np.nan) for row in history], label="val")
        plt.plot(epochs, [row.get(metric_test, np.nan) for row in history], label="test")
        plt.xlabel("epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=160)
        plt.close()

    plot_metric("train_loss", "val_loss", "test_loss", "Loss", "loss_curve.png")
    plot_metric("train_macro_auroc", "val_macro_auroc", "test_macro_auroc", "Macro AUROC", "macro_auroc_curve.png")
    plot_metric("train_macro_auprc", "val_macro_auprc", "test_macro_auprc", "Macro AUPRC", "macro_auprc_curve.png")
    plot_metric("train_macro_f1", "val_macro_f1", "test_macro_f1", "Macro F1 @ 0.5", "macro_f1_curve.png")


def monitor_value(metrics: Dict[str, float], monitor: str) -> float:
    if monitor == "loss":
        return -metrics["loss"]
    if monitor == "auroc":
        return metrics["macro_auroc"]
    if monitor == "auprc":
        return metrics["macro_auprc"]
    raise ValueError(monitor)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
    mean: np.ndarray,
    std: np.ndarray,
    history: List[Dict],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "args": vars(args),
            "mean": mean,
            "std": std,
            "labels": LABELS,
            "history": history,
        },
        path,
    )


def export_integer_model(model: IntegerTCN, path: Path, mean: np.ndarray, std: np.ndarray, args: argparse.Namespace) -> None:
    payload = model.export_integer_state()
    payload["normalization"] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "input_clip": float(args.input_clip),
    }
    payload["config"] = vars(args)

    torch.save(payload, path)

    meta = {
        "format": payload["format"],
        "labels": payload["labels"],
        "num_layers": len(payload["layers"]),
        "normalization": payload["normalization"],
        "config": payload["config"],
    }
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)