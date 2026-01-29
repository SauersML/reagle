#!/usr/bin/env python3
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_overall_metrics(beagle, reagle, out_dir):
    keys = [
        ("unphased_concordance", "Concordance"),
        ("r_squared", "R²"),
        ("iqs", "IQS"),
        ("nonref_concordance", "Non-ref conc"),
    ]
    labels = [k[1] for k in keys]
    b_vals = [beagle.get(k[0], 0.0) if beagle else 0.0 for k in keys]
    r_vals = [reagle.get(k[0], 0.0) if reagle else 0.0 for k in keys]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], b_vals, width, label="Beagle")
    ax.bar([i + width / 2 for i in x], r_vals, width, label="Reagle")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Accuracy Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overall_metrics.png"), dpi=150)
    plt.close(fig)

def plot_per_class_accuracy(beagle, reagle, out_dir):
    keys = [
        ("homref_accuracy", "HomRef"),
        ("het_accuracy", "Het"),
        ("homalt_accuracy", "HomAlt"),
    ]
    labels = [k[1] for k in keys]
    b_vals = [beagle.get(k[0], 0.0) if beagle else 0.0 for k in keys]
    r_vals = [reagle.get(k[0], 0.0) if reagle else 0.0 for k in keys]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], b_vals, width, label="Beagle")
    ax.bar([i + width / 2 for i in x], r_vals, width, label="Reagle")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_maf_curves(beagle, reagle, out_dir, key, title, filename):
    if not beagle or not reagle:
        return
    b_bins = beagle.get("by_maf", {})
    r_bins = reagle.get("by_maf", {})
    bins = list(b_bins.keys())
    if not bins:
        return

    b_vals = [b_bins[b].get(key, None) for b in bins]
    r_vals = [r_bins.get(b, {}).get(key, None) for b in bins]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bins, b_vals, marker="o", label="Beagle")
    ax.plot(bins, r_vals, marker="o", label="Reagle")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.set_xticklabels(bins, rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)

def plot_confusion_matrix(metrics, title, filename, out_dir):
    if not metrics:
        return
    cm = metrics.get("confusion_matrix")
    if not cm:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["HomRef", "Het", "HomAlt"])
    ax.set_yticklabels(["HomRef", "Het", "HomAlt"])
    ax.set_xlabel("Imputed")
    ax.set_ylabel("Truth")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center", color="black", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)

def plot_sample_r2_distribution(beagle, reagle, out_dir):
    if not beagle or not reagle:
        return
    b_mean = beagle.get("sample_r2_mean", None)
    r_mean = reagle.get("sample_r2_mean", None)
    b_min = beagle.get("sample_r2_min", None)
    r_min = reagle.get("sample_r2_min", None)
    if b_mean is None or r_mean is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Beagle mean", "Reagle mean"], [b_mean, r_mean], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²")
    ax.set_title("Per-sample R² (mean)")
    ax2 = ax.twinx()
    if b_min is not None and r_min is not None:
        ax2.plot([0, 1], [b_min, r_min], marker="o", color="black", linestyle="--")
        ax2.set_ylabel("R² min")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_r2_summary.png"), dpi=150)
    plt.close(fig)

def plot_ds_calibration(beagle, reagle, out_dir):
    if not beagle or not reagle:
        return
    b_cal = beagle.get("ds_calibration", [])
    r_cal = reagle.get("ds_calibration", [])
    if not b_cal or not r_cal:
        return

    def extract(cal):
        x = []
        y = []
        for row in cal:
            if row.get("mean_pred") is None or row.get("mean_truth") is None:
                continue
            x.append(row["mean_pred"])
            y.append(row["mean_truth"])
        return x, y

    b_x, b_y = extract(b_cal)
    r_x, r_y = extract(r_cal)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 2], [0, 2], color="gray", linestyle="--", label="Ideal")
    ax.plot(b_x, b_y, marker="o", label="Beagle")
    ax.plot(r_x, r_y, marker="o", label="Reagle")
    ax.set_xlabel("Mean predicted dosage")
    ax.set_ylabel("Mean truth dosage")
    ax.set_title("Dosage Calibration")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dosage_calibration.png"), dpi=150)
    plt.close(fig)


def main():
    data_dir = Path("tests/data")
    beagle = load_metrics(data_dir / "beagle_metrics.json")
    reagle = load_metrics(data_dir / "reagle_metrics.json")
    out_dir = data_dir / "plots"
    ensure_dir(out_dir)

    if not beagle or not reagle:
        print("Missing metrics JSON; nothing to plot.")
        return 0

    plot_overall_metrics(beagle, reagle, out_dir)
    plot_per_class_accuracy(beagle, reagle, out_dir)
    plot_maf_curves(beagle, reagle, out_dir, "r_squared", "R² by MAF bin", "r2_by_maf.png")
    plot_maf_curves(beagle, reagle, out_dir, "iqs", "IQS by MAF bin", "iqs_by_maf.png")
    plot_maf_curves(beagle, reagle, out_dir, "f1_score", "F1 by MAF bin", "f1_by_maf.png")
    plot_maf_curves(beagle, reagle, out_dir, "nonref_concordance", "Non-ref Concordance by MAF bin", "nonref_conc_by_maf.png")
    plot_maf_curves(beagle, reagle, out_dir, "switch_error_rate", "Switch Error Rate by MAF bin", "switch_by_maf.png")
    plot_ds_calibration(beagle, reagle, out_dir)
    plot_confusion_matrix(beagle, "Beagle Confusion Matrix", "confusion_beagle.png", out_dir)
    plot_confusion_matrix(reagle, "Reagle Confusion Matrix", "confusion_reagle.png", out_dir)
    plot_sample_r2_distribution(beagle, reagle, out_dir)
    print(f"Plots saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
