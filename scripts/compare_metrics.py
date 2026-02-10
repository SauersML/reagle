#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def is_numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_numeric_values(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_numeric_values(value[key], next_prefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            out.update(flatten_numeric_values(item, next_prefix))
    elif is_numeric(value) and math.isfinite(float(value)):
        out[prefix] = float(value)
    return out


def format_number(value):
    if value is None:
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    rounded = round(value)
    if abs(value - rounded) < 1e-12 and abs(rounded) >= 1000:
        return f"{int(rounded):,}"
    if abs(value) >= 1000:
        return f"{value:,.3f}"
    if abs(value) >= 1:
        return f"{value:.6f}"
    if value == 0:
        return "0"
    return f"{value:.6g}"


def compare_metrics(reagle_metrics, beagle_metrics):
    reagle_flat = flatten_numeric_values(reagle_metrics)
    beagle_flat = flatten_numeric_values(beagle_metrics)

    shared_keys = sorted(set(reagle_flat.keys()) & set(beagle_flat.keys()))
    rows = []
    for key in shared_keys:
        reagle_val = reagle_flat[key]
        beagle_val = beagle_flat[key]
        rows.append((key, reagle_val, beagle_val, reagle_val - beagle_val))

    return rows, len(reagle_flat), len(beagle_flat)


def build_markdown(label, rows, reagle_total, beagle_total):
    lines = []
    lines.append(f"### Full Metrics Comparison: {label}")
    lines.append("")
    lines.append(
        f"Shared numeric metrics compared: **{len(rows)}** "
        f"(Reagle numeric metrics: {reagle_total}, Beagle numeric metrics: {beagle_total})"
    )
    lines.append("")
    lines.append("| Metric | Reagle | Beagle | Delta (Reagle - Beagle) |")
    lines.append("|---|---:|---:|---:|")

    for key, reagle_val, beagle_val, delta in rows:
        lines.append(
            f"| `{key}` | {format_number(reagle_val)} | {format_number(beagle_val)} | {format_number(delta)} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare all shared numeric metrics between Reagle and Beagle JSON outputs.")
    parser.add_argument("--reagle", required=True, help="Path to reagle_metrics.json")
    parser.add_argument("--beagle", required=True, help="Path to beagle_metrics.json")
    parser.add_argument("--label", default="metrics", help="Label shown in output heading")
    parser.add_argument("--output", default="", help="Optional markdown output path")
    args = parser.parse_args()

    reagle_path = Path(args.reagle)
    beagle_path = Path(args.beagle)

    with reagle_path.open("r") as f:
        reagle_metrics = json.load(f)
    with beagle_path.open("r") as f:
        beagle_metrics = json.load(f)

    rows, reagle_total, beagle_total = compare_metrics(reagle_metrics, beagle_metrics)
    markdown = build_markdown(args.label, rows, reagle_total, beagle_total)

    print(markdown)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown)


if __name__ == "__main__":
    main()
