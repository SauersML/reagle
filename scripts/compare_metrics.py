#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

SELECTED_METRICS = [
    ("Accuracy", "Dosage R^2", "r_squared"),
    ("Accuracy", "IQS", "iqs"),
    ("Accuracy", "INFO score (approx)", "info_score_approx"),
    ("Accuracy", "Unphased concordance", "unphased_concordance"),
    ("Accuracy", "Non-ref concordance", "nonref_concordance"),
    ("Accuracy", "Precision", "precision"),
    ("Accuracy", "Recall", "recall"),
    ("Accuracy", "F1 score", "f1_score"),
    ("Accuracy", "SEN mean", "sen_mean"),
    ("Phasing", "Switch error rate", "switch_error_rate"),
    ("Phasing", "Phase concordance", "phase_concordance"),
    ("Phasing", "N50 phase block", "n50_phase_block"),
    ("Runtime", "Calculation time (sec)", "calculation_time_sec"),
    ("Runtime", "AF time (sec)", "af_time_sec"),
    ("Runtime", "Diagnostics time (sec)", "diagnostic_time_sec"),
    ("Runtime", "Metrics loop time (sec)", "metrics_loop_time_sec"),
    ("Context", "Sites compared", "sites_compared"),
    ("Context", "Total genotypes", "total_genotypes"),
    ("Context", "Non-ref total", "nonref_total"),
]


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
    section_rows = {}

    for section, label, key in SELECTED_METRICS:
        reagle_val = reagle_flat.get(key)
        beagle_val = beagle_flat.get(key)
        if reagle_val is None or beagle_val is None:
            continue

        delta = reagle_val - beagle_val
        section_rows.setdefault(section, []).append((label, key, reagle_val, beagle_val, delta))

    return section_rows, len(shared_keys), len(reagle_flat), len(beagle_flat)


def build_markdown(label, section_rows, shared_count, reagle_total, beagle_total):
    lines = []
    lines.append(f"### Metrics Comparison: {label}")
    lines.append("")
    lines.append(
        f"Shared numeric metrics compared: **{shared_count}** "
        f"(Reagle numeric metrics: {reagle_total}, Beagle numeric metrics: {beagle_total})"
    )
    lines.append("")

    section_order = ["Accuracy", "Phasing", "Runtime", "Context"]
    for section in section_order:
        rows = section_rows.get(section, [])
        if not rows:
            continue
        lines.append(f"#### {section}")
        lines.append("")
        lines.append("| Metric | Key | Reagle | Beagle | Delta (Reagle - Beagle) |")
        lines.append("|---|---|---:|---:|---:|")
        for metric_label, metric_key, reagle_val, beagle_val, delta in rows:
            lines.append(
                f"| {metric_label} | `{metric_key}` | {format_number(reagle_val)} | {format_number(beagle_val)} | {format_number(delta)} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare Reagle and Beagle on a curated set of useful metrics.")
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

    section_rows, shared_count, reagle_total, beagle_total = compare_metrics(reagle_metrics, beagle_metrics)
    markdown = build_markdown(args.label, section_rows, shared_count, reagle_total, beagle_total)

    print(markdown)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown)


if __name__ == "__main__":
    main()
