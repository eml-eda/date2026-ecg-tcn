import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary

from tcn.constants import LABELS
from tcn.data import build_loaders, build_splits, load_ptbxl
from tcn.model_pruning import PrunableTCN, count_parameters
from tcn.training import (
    compute_pos_weight,
    evaluate,
    make_scheduler,
    monitor_value,
    safe_classification_metrics,
    save_history_csv,
    save_plots,
    set_seed,
)

from plinio.methods import PIT

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="runs_pruning")
    parser.add_argument("--sampling-rate", type=int, choices=[100, 500], default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--monitor", type=str, choices=["auroc", "auprc", "loss"], default="auroc")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--input-clip", type=float, default=8.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dilations", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--pit-arch-lr", type=float, default=1e-3)
    parser.add_argument("--pit-arch-weight-decay", type=float, default=0.0)
    parser.add_argument("--pit-reg-strength", type=float, default=1e-8)
    parser.add_argument("--pit-arch-start-epoch", type=int, default=1)
    parser.add_argument("--pit-discrete-cost", type=bool, default=True)
    return parser.parse_args()


def print_network_info(model: nn.Module, input_shape: torch.Size) -> None:
    print(
        summary(
            model,
            input_size=(1, *input_shape),
            col_names=("input_size", "output_size", "num_params"),
            depth=8,
            verbose=1,
            device=str(next(model.parameters()).device),
        )
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    early_stop_split = "val"

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dilations = [int(x) for x in args.dilations.split(",") if x.strip()]
    if not dilations:
        raise ValueError("No dilations provided")

    print("Loading PTB-XL...")
    signals, labels, metadata = load_ptbxl(
        root=root,
        sampling_rate=args.sampling_rate,
        use_cache=args.cache_dataset,
        max_records=args.max_records,
    )
    print(f"Loaded records: {len(metadata)}")
    print(f"Signal tensor shape: {signals.shape} (N, C, T)")

    loaders, mean, std = build_loaders(
        signals=signals,
        labels=labels,
        metadata=metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_clip=args.input_clip,
    )
    splits = build_splits(metadata)

    device = torch.device(args.device)
    print("Using device:", device)

    dense_model = PrunableTCN(
        in_channels=12,
        num_classes=len(LABELS),
        channels=args.channels,
        kernel_size=args.kernel_size,
        dilations=dilations,
        dropout=args.dropout,
    ).to(device)
    dense_params = count_parameters(dense_model)
    print_network_info(dense_model, signals.shape[1:])

    receptive_field = dense_model.receptive_field()
    seq_len = int(signals.shape[-1])
    print(f"Model receptive field: {receptive_field} samples")
    if receptive_field < seq_len:
        print(
            "WARNING: receptive field is shorter than the input sequence. "
            "With the last-timestep head, the classifier will not see the full record."
        )
        print(
            f"         seq_len={seq_len}, receptive_field={receptive_field}. "
            "Increase dilations and/or kernel size for full-window coverage."
        )

    model = PIT(
        dense_model,
        input_shape=tuple(signals.shape[1:]),
        discrete_cost=args.pit_discrete_cost,
    ).to(device)

    pos_weight = compute_pos_weight(labels, splits.train_idx).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    net_optimizer = torch.optim.AdamW(model.net_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    arch_optimizer = torch.optim.AdamW(
        model.nas_parameters(),
        lr=args.pit_arch_lr,
        weight_decay=args.pit_arch_weight_decay,
    )
    net_scheduler = make_scheduler(net_optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)
    arch_scheduler = make_scheduler(arch_optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    history: List[Dict[str, Any]] = []
    best_metric = -float("inf")
    best_epoch = -1
    patience_counter = 0
    best_pruning_summary: Dict[str, Any] | None = None

    print(f"Dense parameter count: {dense_params}")
    print(f"Initial PIT cost: {float(model.cost.detach().cpu().item()):.2f}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        update_architecture = epoch >= args.pit_arch_start_epoch
        if update_architecture:
            model.train_net_and_nas()
        else:
            model.train_net_only()

        train_metrics = train_one_epoch_pit(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            net_optimizer=net_optimizer,
            arch_optimizer=arch_optimizer,
            device=device,
            reg_strength=args.pit_reg_strength,
            update_architecture=update_architecture,
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        test_metrics = evaluate(model, loaders["test"], criterion, device)

        net_scheduler.step()
        arch_scheduler.step()

        exported_model = model.export()
        exported_params = count_parameters(exported_model)
        pit_summary = model.summary()

        row = {
            "epoch": epoch,
            "lr": net_optimizer.param_groups[0]["lr"],
            "pit_arch_lr": arch_optimizer.param_groups[0]["lr"],
            "pit_architecture_updates": int(update_architecture),
            "pit_cost": train_metrics["pit_cost"],
            "pit_reg_loss": train_metrics["reg_loss"],
            "exported_parameters": exported_params,
            "parameter_reduction": 1.0 - (exported_params / max(dense_params, 1)),
            "train_loss": train_metrics["loss"],
            "train_task_loss": train_metrics["task_loss"],
            "train_macro_auroc": train_metrics["macro_auroc"],
            "train_macro_auprc": train_metrics["macro_auprc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_loss": test_metrics["loss"],
            "test_macro_auroc": test_metrics["macro_auroc"],
            "test_macro_auprc": test_metrics["macro_auprc"],
            "test_macro_f1": test_metrics["macro_f1"],
            "epoch_time_sec": time.time() - t0,
        }
        history.append(row)

        current_metric = monitor_value(val_metrics, args.monitor)
        improved = current_metric > best_metric

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val AUROC {val_metrics['macro_auroc']:.4f} | "
            f"test AUROC {test_metrics['macro_auroc']:.4f} | "
            f"PIT cost {train_metrics['pit_cost']:.2f} | "
            f"exported params {exported_params} | "
            f"monitor({early_stop_split}/{args.monitor})={current_metric:.4f}"
        )

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            save_pit_checkpoint(
                outdir / "best_pit_search.pt",
                model,
                net_optimizer,
                arch_optimizer,
                net_scheduler,
                arch_scheduler,
                epoch,
                best_metric,
                args,
                mean,
                std,
                history,
            )
            best_pruning_summary = save_pruned_artifacts(outdir, model, dense_params, args)
            with open(outdir / "best_pit_architecture.json", "w") as handle:
                json.dump(pit_summary, handle, indent=2)
        else:
            patience_counter += 1

        if epoch % args.save_every == 0:
            save_pit_checkpoint(
                outdir / f"pit_checkpoint_epoch_{epoch:03d}.pt",
                model,
                net_optimizer,
                arch_optimizer,
                net_scheduler,
                arch_scheduler,
                epoch,
                best_metric,
                args,
                mean,
                std,
                history,
            )

        save_history_csv(history, outdir / "history.csv")
        with open(outdir / "history.json", "w") as handle:
            json.dump(history, handle, indent=2)
        save_plots(history, outdir)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    print(f"Best epoch: {best_epoch}")
    print(f"Best monitor value: {best_metric:.6f}")

    summary_payload: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "monitor": args.monitor,
        "early_stop_split": early_stop_split,
        "sampling_rate": args.sampling_rate,
        "labels": LABELS,
        "dense_parameters": dense_params,
    }
    if best_pruning_summary is not None:
        summary_payload["best_pruning"] = {
            "exported_parameters": best_pruning_summary["exported_parameters"],
            "parameter_reduction": best_pruning_summary["parameter_reduction"],
            "pit_cost": best_pruning_summary["pit_cost"],
        }
    with open(outdir / "summary.json", "w") as handle:
        json.dump(summary_payload, handle, indent=2)


def save_pit_checkpoint(
    path: Path,
    model: nn.Module,
    net_optimizer: torch.optim.Optimizer,
    arch_optimizer: torch.optim.Optimizer,
    net_scheduler,
    arch_scheduler,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
    mean,
    std,
    history: List[Dict[str, Any]],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "net_optimizer_state": net_optimizer.state_dict(),
            "arch_optimizer_state": arch_optimizer.state_dict(),
            "net_scheduler_state": net_scheduler.state_dict() if net_scheduler is not None else None,
            "arch_scheduler_state": arch_scheduler.state_dict() if arch_scheduler is not None else None,
            "best_metric": best_metric,
            "args": vars(args),
            "mean": mean,
            "std": std,
            "labels": LABELS,
            "history": history,
            "pit_summary": model.summary(),
        },
        path,
    )


def train_one_epoch_pit(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    net_optimizer: torch.optim.Optimizer,
    arch_optimizer: torch.optim.Optimizer,
    device: torch.device,
    reg_strength: float,
    update_architecture: bool,
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    task_losses: List[float] = []
    reg_losses: List[float] = []
    costs: List[float] = []
    probs_all = []
    targets_all = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        net_optimizer.zero_grad(set_to_none=True)
        arch_optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        task_loss = criterion(logits, y)
        cost = model.cost if reg_strength > 0.0 else torch.zeros((), device=device)
        reg_loss = reg_strength * cost if update_architecture else torch.zeros((), device=device)
        loss = task_loss + reg_loss

        loss.backward()
        net_optimizer.step()
        if update_architecture:
            arch_optimizer.step()

        losses.append(float(loss.item()))
        task_losses.append(float(task_loss.item()))
        reg_losses.append(float(reg_loss.item()))
        costs.append(float(cost.detach().item()))
        probs_all.append(torch.sigmoid(logits.detach()).cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

    y_true = np.concatenate(targets_all, axis=0)
    y_prob = np.concatenate(probs_all, axis=0)
    metrics = safe_classification_metrics(y_true, y_prob)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    metrics["task_loss"] = float(sum(task_losses) / max(len(task_losses), 1))
    metrics["reg_loss"] = float(sum(reg_losses) / max(len(reg_losses), 1))
    metrics["pit_cost"] = float(sum(costs) / max(len(costs), 1))
    return metrics


def summarize_pruning(pit_model: nn.Module, exported_model: nn.Module, dense_params: int) -> Dict[str, Any]:
    arch_summary = pit_model.summary()
    pruned_params = count_parameters(exported_model)
    return {
        "dense_parameters": dense_params,
        "exported_parameters": pruned_params,
        "parameter_reduction": 1.0 - (pruned_params / max(dense_params, 1)),
        "pit_cost": float(pit_model.cost.detach().cpu().item()),
        "architecture": arch_summary,
    }


def save_pruned_artifacts(
    outdir: Path,
    pit_model: nn.Module,
    dense_params: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    exported_model = pit_model.export()
    pruning_summary = summarize_pruning(pit_model, exported_model, dense_params)
    pruning_summary["config"] = vars(args)

    # torch.save(exported_model, outdir / "best_pruned_model.pt")
    # torch.save(
    #     {
    #         "state_dict": exported_model.state_dict(),
    #         "summary": pruning_summary,
    #     },
    #     outdir / "best_pruned_state.pt",
    # )
    with open(outdir / "best_pruning_summary.json", "w") as handle:
        json.dump(pruning_summary, handle, indent=2)
    return pruning_summary


if __name__ == "__main__":
    main()
