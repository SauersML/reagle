#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_float(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(run_dir: Path):
    rows = []
    for tsv in sorted(run_dir.glob("exp-*/chr21_fast_metrics.tsv")):
        exp_dir = tsv.parent
        with tsv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("tool") != "reagle":
                    continue
                rows.append(
                    {
                        "exp_dir": str(exp_dir),
                        "runtime_sec": parse_float(row.get("runtime_sec")),
                        "r2": parse_float(row.get("r_squared")),
                        "iqs": parse_float(row.get("iqs")),
                        "hellinger": parse_float(row.get("hellinger_score")),
                        "switch": parse_float(row.get("switch_error_rate")),
                        "phase_conc": parse_float(row.get("phase_concordance")),
                    }
                )
    return rows


def fmt(x):
    return "NA" if x is None else f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser(description="Summarize chr21 fast metrics from experiment runner output.")
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    rows = load_rows(args.run_dir)
    if not rows:
        print(f"No reagle rows found under {args.run_dir}/exp-*/chr21_fast_metrics.tsv")
        return 1

    rows.sort(
        key=lambda r: (
            -1.0 if r["r2"] is None else -r["r2"],
            9e9 if r["switch"] is None else r["switch"],
            9e9 if r["hellinger"] is None else r["hellinger"],
        )
    )

    print("rank\texp_dir\truntime_sec\tr2\tiqs\tswitch_error\thellinger\tphase_concordance")
    for i, r in enumerate(rows, start=1):
        print(
            f"{i}\t{r['exp_dir']}\t{fmt(r['runtime_sec'])}\t{fmt(r['r2'])}\t{fmt(r['iqs'])}\t{fmt(r['switch'])}\t{fmt(r['hellinger'])}\t{fmt(r['phase_conc'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
