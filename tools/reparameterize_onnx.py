#!/usr/bin/env python3
"""Apply guarded FRAPPE deployment reparameterizations to an ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compressors.frappe.harness.onnx_reparameterization import (
    RewriteError,
    fold_encoder_input_normalization,
    fold_fixed_prefix_affine,
    fuse_tanh_gelu,
    op_counts,
    replace_nonoverlap_convtranspose,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite only the proven FRAPPE ONNX deployment patterns."
    )
    parser.add_argument("--input", type=Path, required=True, help="source ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="new ONNX model")
    parser.add_argument("--report", type=Path, help="JSON report (default: <output>.report.json)")
    parser.add_argument("--fold-input-normalization", action="store_true")
    parser.add_argument("--fold-fixed-prefix", action="store_true")
    parser.add_argument(
        "--prefix-conv-name", help="restrict fixed-prefix folding to this Conv name"
    )
    parser.add_argument("--fuse-tanh-gelu", action="store_true")
    parser.add_argument("--replace-nonoverlap-convtranspose", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_match(changed: int, pattern: str) -> None:
    """Refuse a requested transformation instead of silently copying the graph."""
    if changed == 0:
        raise RewriteError(f"no supported {pattern} pattern matched")


def main() -> None:
    args = parse_args()
    enabled = [
        args.fold_input_normalization,
        args.fold_fixed_prefix,
        args.fuse_tanh_gelu,
        args.replace_nonoverlap_convtranspose,
    ]
    if not any(enabled):
        raise SystemExit("select at least one rewrite flag")
    if not args.input.is_file():
        raise SystemExit(f"input model does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("refusing to overwrite the source model; choose a distinct --output")

    import onnx

    model = onnx.load(args.input)
    source_counts = op_counts(model)
    transformations = []
    try:
        if args.fold_input_normalization:
            result = fold_encoder_input_normalization(model)
            require_match(result.changed, "encoder input-normalization")
            model = result.model
            transformations.append({"name": "fold_input_normalization", **result.as_dict()})
        if args.fold_fixed_prefix:
            result = fold_fixed_prefix_affine(model, conv_name=args.prefix_conv_name)
            require_match(result.changed, "fixed-prefix affine")
            model = result.model
            transformations.append({"name": "fold_fixed_prefix", **result.as_dict()})
        if args.fuse_tanh_gelu:
            result = fuse_tanh_gelu(model)
            require_match(result.changed, "expanded tanh-GELU")
            model = result.model
            transformations.append({"name": "fuse_tanh_gelu", **result.as_dict()})
        if args.replace_nonoverlap_convtranspose:
            result = replace_nonoverlap_convtranspose(model)
            require_match(result.changed, "non-overlapping ConvTranspose")
            model = result.model
            transformations.append({"name": "replace_nonoverlap_convtranspose", **result.as_dict()})
    except RewriteError as error:
        raise SystemExit(f"rewrite refused: {error}") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    onnx.save(model, temporary)
    temporary.replace(args.output)
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_op_counts": source_counts,
        "output_op_counts": op_counts(model),
        "transformations": transformations,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
