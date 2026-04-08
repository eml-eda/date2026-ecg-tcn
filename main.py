import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torchinfo import summary


from tcn.cli import parse_args
from tcn.constants import (
    LABELS,
)
from tcn.data import (
    build_loaders,
    build_splits,
    load_ptbxl,
)
from tcn.model import (
    IntegerTCN,
    unwrap_model,
)
from tcn.training import (
    compute_pos_weight,
    evaluate,
    export_integer_model,
    make_scheduler,
    monitor_value,
    save_checkpoint,
    save_history_csv,
    save_plots,
    set_seed,
    train_one_epoch,
)


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

    freeze_quant_epoch = args.freeze_quant_epoch
    if freeze_quant_epoch < 0:
        freeze_quant_epoch = max(1, int(round(0.6 * args.epochs)))

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
    model = IntegerTCN(
        in_channels=12,
        num_classes=len(LABELS),
        channels=args.channels,
        kernel_size=args.kernel_size,
        dilations=dilations,
        dropout=args.dropout,
        causal=True,
        observer_momentum=args.observer_momentum,
        pow2_scales=args.pow2_scales,
    ).to(device)
    print_network_info(model, signals.shape[1:])

    receptive_field = unwrap_model(model).receptive_field()
    seq_len = int(signals.shape[-1])
    print(f"Model receptive field: {receptive_field} samples")
    if receptive_field < seq_len:
        print(
            "WARNING: receptive field is shorter than the input sequence. "
            "With the streaming last-timestep head, the classifier will not see the full record."
        )
        print(
            f"         seq_len={seq_len}, receptive_field={receptive_field}. "
            "Increase dilations and/or kernel size for full-window coverage."
        )

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed, continuing without it: {e}")

    pos_weight = compute_pos_weight(labels, splits.train_idx).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    history: List[Dict] = []
    best_metric = -float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch == freeze_quant_epoch:
            base_model = unwrap_model(model)
            if hasattr(base_model, "freeze_observers"):
                base_model.freeze_observers()
            print(f"Frozen quantization observers at epoch {epoch}")

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

        stop_split_metrics = val_metrics
        current_metric = monitor_value(stop_split_metrics, args.monitor)
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
            save_checkpoint(
                outdir / "best_model.pt",
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
            export_integer_model(unwrap_model(model), outdir / "best_integer_export.pt", mean, std, args)
        else:
            patience_counter += 1

        if epoch % args.save_every == 0:
            save_checkpoint(
                outdir / f"checkpoint_epoch_{epoch:03d}.pt",
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
        with open(outdir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        save_plots(history, outdir)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    print(f"Best epoch: {best_epoch}")
    print(f"Best monitor value: {best_metric:.6f}")

    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "monitor": args.monitor,
        "early_stop_split": early_stop_split,
        "sampling_rate": args.sampling_rate,
        "labels": LABELS,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
