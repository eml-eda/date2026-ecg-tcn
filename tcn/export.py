#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


INT32_QMAX = 2**31 - 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("app/model.h"))
    p.add_argument("--prefix", type=str, default="model")
    p.add_argument("--shift", type=int, default=24)
    return p.parse_args()


def sanitize_ident(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        ok = ch == "_" or ch.isalnum()
        if not ok:
            out.append("_")
        elif i == 0 and ch.isdigit():
            out.append("_")
            out.append(ch)
        else:
            out.append(ch)
    ident = "".join(out).strip("_")
    return ident or "model"


def torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_dilations(raw: Any) -> List[int]:
    if isinstance(raw, str):
        return [int(x) for x in raw.split(",") if x.strip()]
    if isinstance(raw, Sequence):
        return [int(x) for x in raw]
    raise ValueError(f"Unsupported dilations value: {raw!r}")


def strip_orig_mod_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return state
    if not all(isinstance(k, str) for k in state.keys()):
        return state
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in state.items()
    }


def build_payload_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import main as main
    except Exception as exc:
        raise RuntimeError(
            "Failed to import train.py dependencies while loading a training checkpoint. "
            "If possible, export from best_integer_export.pt instead."
        ) from exc

    args = dict(checkpoint["args"])
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    labels = list(checkpoint.get("labels", [])) or list(main.LABELS)

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
    state = strip_orig_mod_prefix(checkpoint["model_state"])
    model.load_state_dict(state)
    model.eval()

    payload = model.export_integer_state()
    payload["normalization"] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "input_clip": float(args["input_clip"]),
    }
    payload["config"] = args
    payload["labels"] = labels
    return payload


def load_payload(path: Path) -> Dict[str, Any]:
    blob = torch_load(path)
    if not isinstance(blob, dict):
        raise TypeError(f"Unsupported payload type in {path}: {type(blob)!r}")

    if "format" in blob and "layers" in blob:
        return blob
    if "model_state" in blob and "args" in blob:
        return build_payload_from_checkpoint(blob)

    raise ValueError(
        f"Unsupported file format for {path}. Expected best_model.pt or best_integer_export.pt."
    )


def approx_multiplier(real_multiplier: float, shift: int) -> Tuple[int, int]:
    if not math.isfinite(real_multiplier) or real_multiplier <= 0.0:
        raise ValueError(f"Invalid scale ratio: {real_multiplier}")
    if abs(real_multiplier - 1.0) < 1e-12:
        return 1, 0
    mult = int(round(real_multiplier * float(1 << shift)))
    mult = max(1, min(mult, INT32_QMAX))
    return mult, shift


def make_scale(src_scale: float, dst_scale: float, shift: int) -> Tuple[int, int]:
    return approx_multiplier(float(src_scale) / float(dst_scale), shift)


def layer_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(layer["name"]): layer for layer in payload["layers"]}


def block_ids(layers: Dict[str, Dict[str, Any]]) -> List[int]:
    ids = set()
    for name in layers:
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks":
            ids.add(int(parts[1]))
    return sorted(ids)


def get_layer(
    layers: Dict[str, Dict[str, Any]],
    name: str,
    expected_type: str | None = None,
) -> Dict[str, Any]:
    if name not in layers:
        raise KeyError(f"Missing layer {name!r} in integer export")
    layer = layers[name]
    if expected_type is not None and layer.get("type") != expected_type:
        raise TypeError(
            f"Layer {name!r} has type {layer.get('type')!r}, expected {expected_type!r}"
        )
    return layer


def arr_i8(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int8)


def arr_i32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int32)


def pack_conv_weight(layer: Dict[str, Any]) -> np.ndarray:
    w = arr_i8(layer["weight_int8"])
    if w.ndim != 3:
        raise ValueError(f"Conv weight for {layer['name']} must have 3 dims, got {w.shape}")
    return np.transpose(w, (0, 2, 1)).copy()


def ensure_bias(layer: Dict[str, Any]) -> np.ndarray:
    out_ch = int(layer["out_channels"])
    if layer.get("bias_int32") is None:
        return np.zeros(out_ch, dtype=np.int32)
    return arr_i32(layer["bias_int32"]).reshape(out_ch)


def ensure_mul(layer: Dict[str, Any]) -> np.ndarray:
    out_ch = int(layer["out_channels"])
    return arr_i32(layer["requant_multiplier_per_out_channel"]).reshape(out_ch)


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
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(vals[i : i + per_line]))
    return ",\n".join(lines)


def emit_array(
    lines: List[str],
    ctype: str,
    name: str,
    values: Iterable[str],
    size: int,
    align: bool = False,
    per_line: int = 12,
) -> None:
    prefix = "MODEL_ALIGN " if align else ""
    lines.append(f"static const {prefix}{ctype} {name}[{size}] = {{")
    body = format_values(values, per_line)
    if body:
        lines.append(body)
    lines.append("};")
    lines.append("")


def emit_float_scalar(lines: List[str], name: str, value: float) -> None:
    lines.append(f"static const float {name} = {c_float(value)};")


def emit_scale_meta(lines: List[str], prefix: str, layer_name: str, layer: Dict[str, Any]) -> None:
    safe = layer_name.replace(".", "_")
    emit_float_scalar(lines, f"{prefix}_{safe}_x_scale", float(layer["x_scale"]))
    emit_float_scalar(lines, f"{prefix}_{safe}_y_scale", float(layer["y_scale"]))
    lines.append("")


def emit_relu_meta(lines: List[str], prefix: str, layer_name: str, layer: Dict[str, Any]) -> None:
    safe = layer_name.replace(".", "_")
    emit_float_scalar(lines, f"{prefix}_{safe}_scale", float(layer["scale"]))
    lines.append("")


def emit_conv_arrays(lines: List[str], prefix: str, name: str, layer: Dict[str, Any]) -> None:
    safe = name.replace(".", "_")
    w = pack_conv_weight(layer).reshape(-1)
    b = ensure_bias(layer)
    m = ensure_mul(layer)
    emit_array(lines, "int8_t", f"{prefix}_{safe}_w", (c_int(v) for v in w), w.size, align=True, per_line=24)
    emit_array(lines, "int32_t", f"{prefix}_{safe}_b", (c_int(v) for v in b), b.size, per_line=8)
    emit_array(lines, "int32_t", f"{prefix}_{safe}_m", (c_int(v) for v in m), m.size, per_line=8)
    emit_scale_meta(lines, prefix, name, layer)


def render_header(payload: Dict[str, Any], src_path: Path, prefix: str, shift: int) -> str:
    lines: List[str] = []
    prefix = sanitize_ident(prefix)
    guard = f"{prefix.upper()}_H_"
    layers = layer_map(payload)
    labels = [str(x) for x in payload["labels"]]
    norm = payload["normalization"]
    cfg = payload["config"]
    stem = get_layer(layers, "stem", "IntegerConv1dQAT")
    stem_relu = get_layer(layers, "stem_relu", "IntegerReLUQAT")
    head = get_layer(layers, "head", "IntegerConv1dQAT")
    ids = block_ids(layers)
    warmup = int(payload["streaming"]["warmup_context_samples"])
    receptive_field = int(payload["streaming"]["receptive_field_samples"])
    in_ch = int(stem["in_channels"])
    ch = int(cfg["channels"])
    n_classes = int(head["out_channels"])

    lines.append("#pragma once")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append('#include "tcn.h"')
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif")
    lines.append("")
    lines.append("#ifndef MODEL_ALIGN")
    lines.append("#if defined(__GNUC__)")
    lines.append('#define MODEL_ALIGN __attribute__((aligned(4)))')
    lines.append("#else")
    lines.append("#define MODEL_ALIGN")
    lines.append("#endif")
    lines.append("#endif")
    lines.append("")
    lines.append(f"// Generated from: {src_path}")
    lines.append(f"// Export format: {payload['format']}")
    lines.append("")
    lines.append(f"#define {prefix.upper()}_IN_CH {in_ch}")
    lines.append(f"#define {prefix.upper()}_CH {ch}")
    lines.append(f"#define {prefix.upper()}_BLOCKS {len(ids)}")
    lines.append(f"#define {prefix.upper()}_CLASSES {n_classes}")
    lines.append(f"#define {prefix.upper()}_WARMUP {warmup}")
    lines.append(f"#define {prefix.upper()}_RECEPTIVE_FIELD {receptive_field}")
    lines.append("")

    emit_array(
        lines,
        "float",
        f"{prefix}_mean",
        (c_float(v) for v in norm["mean"]),
        len(norm["mean"]),
        per_line=8,
    )
    emit_array(
        lines,
        "float",
        f"{prefix}_std",
        (c_float(v) for v in norm["std"]),
        len(norm["std"]),
        per_line=8,
    )
    emit_float_scalar(lines, f"{prefix}_input_clip", float(norm["input_clip"]))
    emit_float_scalar(lines, f"{prefix}_input_scale", float(stem["x_scale"]))
    emit_float_scalar(lines, f"{prefix}_output_scale", float(head["y_scale"]))
    lines.append("")

    emit_array(
        lines,
        "char *const",
        f"{prefix}_labels",
        (c_string(x) for x in labels),
        len(labels),
        per_line=8,
    )

    emit_conv_arrays(lines, prefix, "stem", stem)
    emit_relu_meta(lines, prefix, "stem_relu", stem_relu)

    prev_scale = float(stem_relu["scale"])
    for block_id in ids:
        conv1 = get_layer(layers, f"blocks.{block_id}.conv1", "IntegerConv1dQAT")
        relu1 = get_layer(layers, f"blocks.{block_id}.relu1", "IntegerReLUQAT")
        conv2 = get_layer(layers, f"blocks.{block_id}.conv2", "IntegerConv1dQAT")
        add_relu = get_layer(layers, f"blocks.{block_id}.add_relu", "IntegerAddReLUQAT")
        emit_conv_arrays(lines, prefix, f"blocks.{block_id}.conv1", conv1)
        emit_relu_meta(lines, prefix, f"blocks.{block_id}.relu1", relu1)
        emit_conv_arrays(lines, prefix, f"blocks.{block_id}.conv2", conv2)
        emit_relu_meta(lines, prefix, f"blocks.{block_id}.add_relu", add_relu)
        prev_scale = float(add_relu["scale"])

    emit_conv_arrays(lines, prefix, "head", head)

    prev_scale = float(stem_relu["scale"])
    if ids:
        lines.append(f"static const tcn_block_t {prefix}_blocks[{len(ids)}] = {{")
        for block_id in ids:
            conv1 = get_layer(layers, f"blocks.{block_id}.conv1", "IntegerConv1dQAT")
            relu1 = get_layer(layers, f"blocks.{block_id}.relu1", "IntegerReLUQAT")
            conv2 = get_layer(layers, f"blocks.{block_id}.conv2", "IntegerConv1dQAT")
            add_relu = get_layer(layers, f"blocks.{block_id}.add_relu", "IntegerAddReLUQAT")
            conv1_in_m, conv1_in_r = make_scale(prev_scale, float(conv1["x_scale"]), shift)
            relu1_m, relu1_r = make_scale(float(conv1["y_scale"]), float(relu1["scale"]), shift)
            conv2_in_m, conv2_in_r = make_scale(float(relu1["scale"]), float(conv2["x_scale"]), shift)
            add_x_m, add_x_r = make_scale(float(conv2["y_scale"]), float(add_relu["scale"]), shift)
            add_res_m, add_res_r = make_scale(prev_scale, float(add_relu["scale"]), shift)
            ctx1 = int(conv1["streaming_context_len"])
            ctx2 = int(conv2["streaming_context_len"])
            lines.append("    {")
            lines.append(
                "        .conv1 = {"
                f".in_ch = {int(conv1['in_channels'])}, "
                f".out_ch = {int(conv1['out_channels'])}, "
                f".k = {int(conv1['kernel_size'])}, "
                f".dil = {int(conv1['dilation'])}, "
                f".ctx = {ctx1}, "
                f".in_r = {conv1_in_r}, "
                f".out_r = {int(conv1['requant_shift'])}, "
                f".in_m = {conv1_in_m}, "
                f".w = {prefix}_blocks_{block_id}_conv1_w, "
                f".b = {prefix}_blocks_{block_id}_conv1_b, "
                f".m = {prefix}_blocks_{block_id}_conv1_m"
                "},"
            )
            lines.append(
                "        .relu1 = {"
                f".m = {relu1_m}, .r = {relu1_r}"
                "},"
            )
            lines.append(
                "        .conv2 = {"
                f".in_ch = {int(conv2['in_channels'])}, "
                f".out_ch = {int(conv2['out_channels'])}, "
                f".k = {int(conv2['kernel_size'])}, "
                f".dil = {int(conv2['dilation'])}, "
                f".ctx = {ctx2}, "
                f".in_r = {conv2_in_r}, "
                f".out_r = {int(conv2['requant_shift'])}, "
                f".in_m = {conv2_in_m}, "
                f".w = {prefix}_blocks_{block_id}_conv2_w, "
                f".b = {prefix}_blocks_{block_id}_conv2_b, "
                f".m = {prefix}_blocks_{block_id}_conv2_m"
                "},"
            )
            lines.append(
                "        .add_x = {"
                f".m = {add_x_m}, .r = {add_x_r}"
                "},"
            )
            lines.append(
                "        .add_res = {"
                f".m = {add_res_m}, .r = {add_res_r}"
                "},"
            )
            lines.append("    },")
            prev_scale = float(add_relu["scale"])
        lines.append("};")
    else:
        lines.append(f"static const tcn_block_t *{prefix}_blocks = 0;")
    lines.append("")

    stem_relu_m, stem_relu_r = make_scale(float(stem["y_scale"]), float(stem_relu["scale"]), shift)
    head_in_m, head_in_r = make_scale(prev_scale, float(head["x_scale"]), shift)
    lines.append(f"static const tcn_model_t {prefix} = {{")
    lines.append(f"    .in_ch = {in_ch},")
    lines.append(f"    .ch = {ch},")
    lines.append(f"    .n_blocks = {len(ids)},")
    lines.append(f"    .n_classes = {n_classes},")
    lines.append(f"    .warmup = {warmup},")
    lines.append(
        "    .stem = {"
        f".in_ch = {int(stem['in_channels'])}, "
        f".out_ch = {int(stem['out_channels'])}, "
        f".k = {int(stem['kernel_size'])}, "
        f".dil = {int(stem['dilation'])}, "
        f".ctx = {int(stem['streaming_context_len'])}, "
        f".in_r = 0, "
        f".out_r = {int(stem['requant_shift'])}, "
        f".in_m = 1, "
        f".w = {prefix}_stem_w, "
        f".b = {prefix}_stem_b, "
        f".m = {prefix}_stem_m"
        "},"
    )
    lines.append(
        "    .stem_relu = {"
        f".m = {stem_relu_m}, .r = {stem_relu_r}"
        "},"
    )
    lines.append(f"    .blocks = {prefix}_blocks,")
    lines.append(
        "    .head = {"
        f".in_ch = {int(head['in_channels'])}, "
        f".out_ch = {int(head['out_channels'])}, "
        f".k = {int(head['kernel_size'])}, "
        f".dil = {int(head['dilation'])}, "
        f".ctx = {int(head['streaming_context_len'])}, "
        f".in_r = {head_in_r}, "
        f".out_r = {int(head['requant_shift'])}, "
        f".in_m = {head_in_m}, "
        f".w = {prefix}_head_w, "
        f".b = {prefix}_head_b, "
        f".m = {prefix}_head_m"
        "},"
    )
    lines.append("};")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("} // extern \"C\"")
    lines.append("#endif")
    lines.append("")
    lines.append(f"#endif // {guard}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input)
    header = render_header(payload, args.input, args.prefix, args.shift)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header)
    state_bytes = 0
    for spec in payload["streaming"]["state_spec"]:
        state_bytes += (int(spec["context_len"]) + 1) * int(spec["channels"])
    print(f"Wrote {args.output}")
    print(f"State bytes: {state_bytes}")
    print(f"Blocks: {len(block_ids(layer_map(payload)))}")


if __name__ == "__main__":
    main()
