#!/usr/bin/env python3
"""
Generate microarray-style target/ref VCFs and run a pprof flamegraph.

IMPORTANT: This script does NOT perform dataset size reduction.
It works on whatever input VCFs you provide via --ref and --target.

In the benchmark workflow (.github/workflows/bench.yml):
- The prepare_data job creates a 5% subset of chr22 (~41K markers)
- This script receives those pre-reduced VCFs as input
- The --keep-prob parameter controls additional subsampling (1.0 = no subsampling)

For full-size datasets, use integration_test.py prepare instead of prepare-profile.
"""

import argparse
import gzip
import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path


def run(cmd, *, env=None, timeout=None):
    print(f"CMD: {' '.join(cmd)}")
    if timeout is None or timeout <= 0:
        subprocess.check_call(cmd, env=env)
        return False
    proc = subprocess.Popen(cmd, env=env)
    try:
        proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {timeout}s reached, sending SIGINT...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            print("SIGINT did not stop process; killing.")
            proc.kill()
            proc.communicate()
        return True


def find_default_inputs(repo_root: Path):
    root_ref = repo_root / "ref.vcf.gz"
    root_target = repo_root / "target.vcf.gz"
    if root_ref.exists() and root_target.exists():
        return root_ref, root_target

    return None, None


def find_real_array_file(repo_root: Path):
    candidates = []
    data_dir = repo_root / "data"
    if not data_dir.exists():
        return None, None

    people = ["kat_suricata", "christopher_smith"]
    for person in people:
        person_dir = data_dir / person
        if not person_dir.exists():
            continue
        for name in os.listdir(person_dir):
            lower = name.lower()
            if lower.endswith((".txt", ".csv")) and (
                "23andme" in lower or "ancestry" in lower or "array" in lower
            ):
                candidates.append((person, person_dir / name))

    if not candidates:
        return None, None

    # Prefer 23andMe over other formats if present.
    candidates.sort(key=lambda x: ("23andme" not in x[1].name.lower(), x[1].name))
    return candidates[0]

def infer_person_from_path(path: Path):
    parts = path.parts
    for name in ("kat_suricata", "christopher_smith"):
        if name in parts:
            return name
    return None


def person_data_dir(repo_root: Path, person: str) -> Path:
    return repo_root / "data" / person


def open_vcf(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def build_sparse_positions(vcf_path: Path, out_path: Path, keep_prob: float, seed: int):
    rng = random.Random(seed)
    total = 0
    kept = 0
    with open_vcf(vcf_path) as handle, open(out_path, "w", encoding="utf-8") as out:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            total += 1
            if rng.random() <= keep_prob:
                out.write(f"{fields[0]}\t{fields[1]}\n")
                kept += 1
    return total, kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=Path, default=None, help="Input reference VCF.gz")
    parser.add_argument("--target", type=Path, default=None, help="Input target VCF.gz")
    parser.add_argument("--out-dir", type=Path, default=Path("microarray_profile"), help="Output directory")
    parser.add_argument(
        "--keep-prob",
        type=float,
        default=1.0,
        help="Fraction of markers to keep (default keeps all microarray markers)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for marker sampling")
    parser.add_argument("--chrom", type=str, default=None, help="Optional chrom/region for bcftools -r")
    parser.add_argument("--threads", type=int, default=0, help="Threads for Reagle (0=all)")
    parser.add_argument(
        "--time-limit",
        type=int,
        default=0,
        help="Time limit in seconds for the Reagle run (0 = no limit)",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Generate ref/target from data/* via scripts/prepare_data.py",
    )
    parser.add_argument(
        "--array-file",
        type=Path,
        default=None,
        help="Microarray file (txt/csv) to convert via prepare_data.py array",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ref_vcf, target_vcf = args.ref, args.target

    use_real_data = args.use_real_data
    if not use_real_data and ref_vcf is None and target_vcf is None:
        person, array_file = find_real_array_file(repo_root)
        if array_file:
            print(f"Auto-detected real data for {person}: {array_file}")
            use_real_data = True

    if not use_real_data:
        if ref_vcf is None or target_vcf is None:
            default_ref, default_target = find_default_inputs(repo_root)
            ref_vcf = ref_vcf or default_ref
            target_vcf = target_vcf or default_target

    if shutil.which("bcftools") is None:
        raise SystemExit("bcftools not found on PATH.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref_out = args.out_dir / "ref.vcf.gz"
    ref_for_impute = ref_out
    target_dense = args.out_dir / "target_dense.vcf.gz"
    target_sparse = args.out_dir / "target_sparse.vcf.gz"
    positions = args.out_dir / "target_sparse.positions"

    if use_real_data:
        array_file = args.array_file
        person = None
        if array_file is None:
            person, array_file = find_real_array_file(repo_root)
        else:
            person = infer_person_from_path(array_file)
        if array_file is None or not array_file.exists():
            raise SystemExit(
                "No microarray file found. Provide --array-file or add a 23andme/ancestry "
                "txt/csv under data/kat_suricata or data/christopher_smith."
            )

        prepare = repo_root / "scripts" / "prepare_data.py"
        run(["python3", str(prepare), "reference", str(ref_out)])
        # Microarray sites come from the array conversion (sparse target).
        run(["python3", str(prepare), "array", str(array_file), str(target_sparse)])
        # Dense target represents WGS-like density for the same person.
        if person is None:
            raise SystemExit("Unable to infer person for WGS truth; pass --array-file under data/<person>.")
        truth_source = person_data_dir(repo_root, person)
        run(["python3", str(prepare), "truth", str(truth_source), str(target_dense)])
        # Use WGS-density truth as the reference for imputation in real-data mode.
        ref_wgs = args.out_dir / "ref_wgs.vcf.gz"
        shutil.copy2(target_dense, ref_wgs)
        truth_index = target_dense.with_suffix(target_dense.suffix + ".csi")
        if truth_index.exists():
            shutil.copy2(truth_index, ref_wgs.with_suffix(ref_wgs.suffix + ".csi"))
        else:
            run(["bcftools", "index", "-f", str(ref_wgs)])
        ref_for_impute = ref_wgs
    else:
        if not ref_vcf or not target_vcf or not ref_vcf.exists() or not target_vcf.exists():
            raise SystemExit(
                "Missing input VCFs. Provide --ref and --target, or place ref.vcf.gz + "
                "target.vcf.gz at repo root, or use tests/fixtures/gnomad_hgdp fixtures."
            )

        if args.chrom:
            run(["bcftools", "view", "-r", args.chrom, str(ref_vcf), "-Oz", "-o", str(ref_out)])
            run(["bcftools", "index", "-f", str(ref_out)])
            run(["bcftools", "view", "-r", args.chrom, str(target_vcf), "-Oz", "-o", str(target_dense)])
            run(["bcftools", "index", "-f", str(target_dense)])
        else:
            if ref_out != ref_vcf:
                shutil.copy2(ref_vcf, ref_out)
            if target_dense != target_vcf:
                shutil.copy2(target_vcf, target_dense)

    if args.keep_prob >= 1.0:
        if not use_real_data and target_sparse != target_dense:
            shutil.copy2(target_dense, target_sparse)
            dense_index = target_dense.with_suffix(target_dense.suffix + ".csi")
            if dense_index.exists():
                shutil.copy2(dense_index, target_sparse.with_suffix(target_sparse.suffix + ".csi"))
            else:
                run(["bcftools", "index", "-f", str(target_sparse)])
        if positions.exists():
            positions.unlink()
        print("Sparse positions: kept 100% of microarray markers (no subsampling).")
    else:
        sample_source = target_sparse if use_real_data else target_dense
        total, kept = build_sparse_positions(sample_source, positions, args.keep_prob, args.seed)
        if kept == 0:
            raise SystemExit("Sampling kept 0 markers; increase --keep-prob.")
        print(f"Sparse positions: kept {kept}/{total} markers ({kept/total:.2%})")

        out_path = target_sparse
        if sample_source == target_sparse:
            out_path = target_sparse.with_suffix(".tmp.vcf.gz")
        run(["bcftools", "view", "-R", str(positions), str(sample_source), "-Oz", "-o", str(out_path)])
        if out_path != target_sparse:
            shutil.move(out_path, target_sparse)
        run(["bcftools", "index", "-f", str(target_sparse)])

    run(["cargo", "build", "--release", "--features", "pprof"], env=os.environ.copy())

    env = os.environ.copy()
    env["REAGLE_PPROF"] = "1"
    env["REAGLE_PPROF_OUTPUT"] = str(args.out_dir / "reagle_microarray.svg")

    reagle_bin = repo_root / "target" / "release" / "reagle"
    if not reagle_bin.exists():
        raise SystemExit(f"Reagle binary not found: {reagle_bin}")

    cmd = [
        str(reagle_bin),
        "--ref", str(ref_for_impute),
        "--gt", str(target_sparse),
        "--out", str(args.out_dir / "reagle_microarray"),
        "--gp",
    ]
    threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    cmd.extend(["--nthreads", str(threads)])
    if args.chrom:
        cmd.extend(["--chrom", args.chrom])

    timed_out = run(cmd, env=env, timeout=args.time_limit)

    print("\nDone.")
    if timed_out:
        print("Reagle run timed out; outputs may be partial.")
    print(f"Flamegraph: {env['REAGLE_PPROF_OUTPUT']}")
    txt_path = args.out_dir / "reagle_microarray.txt"
    folded_path = args.out_dir / "reagle_microarray.folded.txt"
    if txt_path.exists():
        print(f"Text profile: {txt_path}")
    else:
        print("Text profile: (missing)")
    if folded_path.exists():
        print(f"Folded profile: {folded_path}")
    else:
        print("Folded profile: (missing)")
    print(f"Sparse target: {target_sparse}")
    print(f"Reference: {ref_for_impute}")


if __name__ == "__main__":
    main()
