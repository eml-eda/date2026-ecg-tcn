#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

import main as main
from tcn.export import load_payload, parse_dilations, sanitize_ident, strip_orig_mod_prefix, torch_load


INT8_QMIN = -128
INT8_QMAX = 127


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("app/test_data.h"))
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--indices", type=str, default="")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--prefix", type=str, default="test_data")
    p.add_argument("--with-logits", action="store_true")
    p.add_argument("--with-raw", action="store_true")
    p.add_argument("--cache-dataset", action="store_true")
    return p.parse_args()


def parse_index_list(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def c_float(x: float) -> str:
    s = f"{float(x):.9g}"
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return s + "f"


def c_int(x: Any) -> str:
    return str(int(x))


def c_string(x: str) -> str:
    return '"' + x.replace("\\", "\\\\").replace('"', '\\"') + '"'


def format_values(values: Iterable[str], per_line: int) -> str:
    vals = list(values)
    if not vals:
        return ""
    out = []
    for i in range(0, len(vals), per_line):
        out.append("    " + ", ".join(vals[i : i + per_line]))
    return ",\n".join(out)


def format_nd_array(arr: np.ndarray, fmt, indent: int = 1) -> str:
    if arr.ndim == 1:
        vals = [fmt(x) for x in arr.tolist()]
        return "    " * indent + "{ " + ", ".join(vals) + " }"

    lines: List[str] = []
    lines.append("    " * indent + "{")
    for i in range(arr.shape[0]):
        tail = "," if i + 1 < arr.shape[0] else ""
        lines.append(format_nd_array(arr[i], fmt, indent + 1) + tail)
    lines.append("    " * indent + "}")
    return "\n".join(lines)


def emit_array(
    lines: List[str],
    ctype: str,
    name: str,
    values: Iterable[str],
    size_expr: str,
    per_line: int = 12,
) -> None:
    lines.append(f"static const {ctype} {name}{size_expr} = {{")
    body = format_values(values, per_line)
    if body:
        lines.append(body)
    lines.append("};")
    lines.append("")


def emit_nd_array(
    lines: List[str],
    ctype: str,
    name: str,
    arr: np.ndarray,
    size_expr: str,
    fmt,
) -> None:
    lines.append(f"static const {ctype} {name}{size_expr} =")
    lines.append(format_nd_array(arr, fmt, indent=0) + ";")
    lines.append("")


def choose_split_indices(
    total: int,
    requested: Sequence[int],
    count: int,
    offset: int,
    shuffle: bool,
    seed: int,
) -> List[int]:
    order = np.arange(total, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)

    if requested:
        chosen = [int(order[i]) for i in requested]
    else:
        lo = max(offset, 0)
        hi = min(lo + max(count, 0), total)
        chosen = [int(x) for x in order[lo:hi]]

    if not chosen:
        raise ValueError("No records selected")
    return chosen


def get_split_pool(metadata) -> np.ndarray:
    splits = main.build_splits(metadata)
    return {
        "train": splits.train_idx,
        "val": splits.val_idx,
        "test": splits.test_idx,
    }


def maybe_load_checkpoint_for_logits(path: Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    blob = torch_load(path)
    if not isinstance(blob, dict) or "model_state" not in blob or "args" not in blob:
        raise ValueError(f"{path} is not a train.py checkpoint")
    return blob


def build_model_from_checkpoint(checkpoint: Dict[str, Any]) -> main.IntegerTCN:
    args = dict(checkpoint["args"])
    labels = list(checkpoint.get("labels", [])) or list(main.LABELS)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    model = main.IntegerTCN(
        in_channels=int(mean.shape[0]),
        num_classes=len(labels),
        channels=int(args["channels"]),
        kernel_size=int(args["kernel_size"]),
        dilations=parse_dilations(args["dilations"]),
        dropout=float(args["dropout"]),
        causal=True,
        observer_momentum=float(args["observer_momentum"]),
        pow2_scales=bool(args["pow2_scales"]),
    )
    model.load_state_dict(strip_orig_mod_prefix(checkpoint["model_state"]))
    model.eval()
    return model


def quantize_input(x_tc: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float, scale: float) -> np.ndarray:
    x = (x_tc - mean[None, :]) / std[None, :]
    x = np.clip(x, -clip, clip)
    xq = np.round(x / scale)
    xq = np.clip(xq, INT8_QMIN, INT8_QMAX)
    return xq.astype(np.int8)


def quantize_input_batch(x_nct: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float, scale: float) -> np.ndarray:
    x = np.array(x_nct, dtype=np.float32, copy=True)
    x -= mean[None, :, None]
    x /= std[None, :, None]
    np.clip(x, -clip, clip, out=x)
    x /= scale
    np.rint(x, out=x)
    np.clip(x, INT8_QMIN, INT8_QMAX, out=x)
    return np.transpose(x.astype(np.int8, copy=False), (0, 2, 1)).copy()


def infer_logits_q(
    model: main.IntegerTCN,
    x_tc: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip: float,
) -> np.ndarray:
    x = (x_tc - mean[None, :]) / std[None, :]
    x = np.clip(x, -clip, clip).astype(np.float32)
    xt = torch.from_numpy(x.T[None, :, :])
    with torch.no_grad():
        y = model.forward_streaming_reference(xt)
    head_scale = float(model.head.act_out_obs.scale.detach().cpu().item())
    y_q = torch.round(y[0] / head_scale).clamp(-32768, 32767).to(torch.int16)
    return y_q.cpu().numpy().astype(np.int16)


def infer_logits_q_batch(
    model: main.IntegerTCN,
    x_nct: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip: float,
) -> np.ndarray:
    x = np.array(x_nct, dtype=np.float32, copy=True)
    x -= mean[None, :, None]
    x /= std[None, :, None]
    np.clip(x, -clip, clip, out=x)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        y = model.forward_streaming_reference(xt)
    head_scale = float(model.head.act_out_obs.scale.detach().cpu().item())
    y_q = torch.round(y / head_scale).clamp(-32768, 32767).to(torch.int16)
    return y_q.cpu().numpy().astype(np.int16)


def render_header(
    prefix: str,
    source_path: Path,
    split_name: str,
    label_names: Sequence[str],
    record_ids: np.ndarray,
    labels: np.ndarray,
    raw_tc: np.ndarray | None,
    xq_tc: np.ndarray,
    logits_q: np.ndarray | None,
) -> str:
    prefix = sanitize_ident(prefix)
    guard = f"{prefix.upper()}_H_"
    n_rec = int(xq_tc.shape[0])
    seq_len = int(xq_tc.shape[1])
    in_ch = int(xq_tc.shape[2])
    n_cls = int(labels.shape[1])

    lines: List[str] = []
    lines.append("#pragma once")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    lines.append("")
    lines.append(f"// Generated from dataset root: {source_path}")
    lines.append(f"// Split: {split_name}")
    lines.append("")
    lines.append(f"#define {prefix.upper()}_NUM_RECORDS {n_rec}")
    lines.append(f"#define {prefix.upper()}_SEQ_LEN {seq_len}")
    lines.append(f"#define {prefix.upper()}_IN_CH {in_ch}")
    lines.append(f"#define {prefix.upper()}_NUM_CLASSES {n_cls}")
    lines.append(f"#define {prefix.upper()}_HAS_LOGITS_Q {1 if logits_q is not None else 0}")
    lines.append("")

    emit_array(
        lines,
        "char *const",
        f"{prefix}_label_names",
        (c_string(x) for x in label_names),
        f"[{n_cls}]",
        per_line=8,
    )
    emit_array(
        lines,
        "uint32_t",
        f"{prefix}_record_ids",
        (c_int(x) for x in record_ids),
        f"[{n_rec}]",
        per_line=8,
    )
    emit_nd_array(lines, "uint8_t", f"{prefix}_labels", labels, f"[{n_rec}][{n_cls}]", c_int)
    if raw_tc is not None:
        emit_nd_array(lines, "float", f"{prefix}_raw", raw_tc, f"[{n_rec}][{seq_len}][{in_ch}]", c_float)
    emit_nd_array(lines, "int8_t", f"{prefix}_xq", xq_tc, f"[{n_rec}][{seq_len}][{in_ch}]", c_int)
    if logits_q is not None:
        emit_nd_array(lines, "int16_t", f"{prefix}_logits_q", logits_q, f"[{n_rec}][{n_cls}]", c_int)

    lines.append("#ifdef __cplusplus")
    lines.append("} // extern \"C\"")
    lines.append("#endif")
    lines.append("")
    lines.append(f"#endif // {guard}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = load_payload(args.model)

    cfg = payload["config"]
    norm = payload["normalization"]
    label_names = list(payload["labels"])
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    input_clip = float(norm["input_clip"])
    input_scale = float(next(layer["x_scale"] for layer in payload["layers"] if layer["name"] == "stem"))
    sampling_rate = int(cfg["sampling_rate"])

    print("Loading PTB-XL...")
    signals, labels, metadata = main.load_ptbxl(
        root=args.root,
        sampling_rate=sampling_rate,
        use_cache=args.cache_dataset,
        max_records=0,
    )

    split_pool = get_split_pool(metadata)[args.split]
    requested = parse_index_list(args.indices)
    rel_idx = choose_split_indices(
        total=len(split_pool),
        requested=requested,
        count=args.count,
        offset=args.offset,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    abs_idx = split_pool[np.asarray(rel_idx, dtype=np.int64)]

    print("Loading selected records into memory...")
    sel_nct = np.asarray(signals[abs_idx], dtype=np.float32)
    label_sel = np.asarray(labels[abs_idx], dtype=np.uint8).copy()
    record_ids = metadata.index.to_numpy()[abs_idx].astype(np.uint32, copy=False)

    print("Quantizing inputs...")
    xq_tc = quantize_input_batch(sel_nct, mean, std, input_clip, input_scale)
    raw_tc = np.transpose(sel_nct, (0, 2, 1)).copy() if (args.with_raw or args.with_logits) else None

    logits_q = None
    ckpt_path = args.checkpoint
    model_ckpt = torch_load(args.model)
    if ckpt_path is None and isinstance(model_ckpt, dict) and "model_state" in model_ckpt:
        ckpt_path = args.model

    if args.with_logits:
        ckpt = maybe_load_checkpoint_for_logits(ckpt_path)
        if ckpt is None:
            raise ValueError(
                "--with-logits requires a train.py checkpoint. "
                "Pass --checkpoint runs/best_model.pt or use --model runs/best_model.pt."
            )
        model = build_model_from_checkpoint(ckpt)
        print("Computing expected logits...")
        logits_q = infer_logits_q_batch(model, sel_nct, mean, std, input_clip)

    print("Formatting C header...")
    header = render_header(
        prefix=args.prefix,
        source_path=args.root,
        split_name=args.split,
        label_names=label_names,
        record_ids=record_ids,
        labels=label_sel,
        raw_tc=raw_tc,
        xq_tc=xq_tc,
        logits_q=logits_q,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Writing output...")
    args.output.write_text(header)

    print(f"Wrote {args.output}")
    print(f"Records: {len(abs_idx)}")
    print(f"Sequence length: {xq_tc.shape[1]}")
    print("ECG IDs:", ", ".join(str(int(x)) for x in record_ids))


if __name__ == "__main__":
    main()
