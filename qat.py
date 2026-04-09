import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torchinfo import summary

from tcn.constants import LABELS
from tcn.data import build_loaders, build_splits, load_ptbxl
from tcn.training import (
    compute_pos_weight,
    evaluate,
    make_scheduler,
    monitor_value,
    save_history_csv,
    save_plots,
    set_seed,
    train_one_epoch,
)

from plinio.methods.mps import MPS, get_default_qinfo, set_pact_clip_values
from plinio.methods.mps.quant.backends import Backend, integerize_arch
from plinio.methods.mps.quant.backends.match.exporter import MATCHExporter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--pruned-model", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="runs_qat")
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
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--qat-weight-bits", type=int, default=8)
    parser.add_argument("--qat-activation-bits", type=int, default=8)

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
    pruned_model_path = Path(args.pruned_model)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    base_model = load_pruned_model(pruned_model_path)
    print_network_info(base_model, signals.shape[1:])

    qinfo = get_default_qinfo((args.qat_weight_bits,), (args.qat_activation_bits,))
    qinfo = set_pact_clip_values(
        base_model,
        torch.rand((1,) + signals.shape[1:]),
        qinfo,
        loaders["val"],
        disable_shared_quantizers=True,
        quantize_output=False,
        use_percentile=True,
        percentile=99.0
    )

    model = MPS(
        base_model,
        input_shape=tuple(signals.shape[1:]),
        qinfo=qinfo,
        disable_shared_quantizers=True,
    ).to(device)

    pos_weight = compute_pos_weight(labels, splits.train_idx).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.net_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    history: List[Dict[str, Any]] = []
    best_metric = -float("inf")
    best_epoch = -1
    patience_counter = 0

    print("Loaded pruned model from:", pruned_model_path)
    print(f"Running fixed-precision MPS QAT at W{args.qat_weight_bits}/A{args.qat_activation_bits}.")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        test_metrics = evaluate(model, loaders["test"], criterion, device)

        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
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
            f"monitor({early_stop_split}/{args.monitor})={current_metric:.4f}"
        )

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            save_qat_checkpoint(
                outdir / "best_qat_search.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                args,
                mean,
                std,
                history,
            )
        else:
            patience_counter += 1

        if epoch % args.save_every == 0:
            save_qat_checkpoint(
                outdir / f"qat_checkpoint_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
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

    # Final ONNX export
    if best_epoch != -1:
        best_checkpoint = torch.load(outdir / "best_qat_search.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state"])
        model.eval()

        # everything needs to be on CPU for export
        exported_model = model.cpu().export()
        exported_model.eval()
        full_int_model = integerize_arch(exported_model.cpu(), Backend.MATCH)
        full_int_model = full_int_model.cpu()
        pos_weight = pos_weight.cpu()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        test_metrics = evaluate(full_int_model, loaders["test"], criterion, 'cpu')
        print(f"Exported full-integer model test AUROC: {test_metrics['macro_auroc']:.4f}")

        onnx_exporter = MATCHExporter()
        onnx_exporter.export(
            network=full_int_model,
            input_shape=torch.Size((1, *signals.shape[1:])),
            path=outdir,
        )


def save_qat_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
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
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "args": vars(args),
            "mean": mean,
            "std": std,
            "labels": LABELS,
            "history": history,
            "mps_summary": model.summary(),
        },
        path,
    )


def load_pruned_model(path: Path) -> nn.Module:
    model = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Expected {path} to contain a serialized nn.Module. "
            "Use the best_pruned_model.pt artifact produced by pruning.py."
        )
    return model


if __name__ == "__main__":
    main()
