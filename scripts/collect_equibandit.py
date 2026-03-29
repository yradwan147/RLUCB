#!/usr/bin/env python3
"""
Collect EquiBandit IBEX results and compute knowledge + Gini statistics.

Usage:
    python scripts/collect_equibandit.py [--results-dir results/equibandit]

Outputs:
    - Console: table-ready mean ± SE for knowledge and Gini coefficient
    - results/equibandit/summary.csv
"""

import argparse
import csv
import numpy as np
from pathlib import Path

CONFIGS = [
    (6,  0.005),
    (6,  0.01),
    (6,  0.05),
    (20, 0.005),
    (20, 0.01),
    (20, 0.05),
]
SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 6283, 9999, 11111]
ALGORITHMS = ["equi_bandit", "bkt_bandit", "random"]
ALG_LABELS  = {"equi_bandit": "EquiBandit", "bkt_bandit": "BKT-Bandit", "random": "Random"}


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of exposure distribution."""
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    x = np.sort(counts)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


def load_seed_result(results_dir: Path, K: int, lam: float, seed: int, alg: str):
    """
    Load final knowledge and quiz-count distribution from CSV output.
    Returns (final_knowledge, gini_coeff) or (None, None) if missing.
    """
    seed_dir = results_dir / f"K{K}_L{lam}" / f"seed{seed}"
    # Try to find a CSV file for the algorithm
    candidates = list(seed_dir.glob(f"*{alg}*.csv")) + list(seed_dir.glob("*.csv"))
    if not candidates:
        return None, None

    # Read all CSVs and find the one belonging to this algorithm
    for csv_path in sorted(candidates):
        try:
            rows = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                continue

            # Check if this CSV belongs to the right algorithm
            if "algorithm" in rows[0] and rows[0]["algorithm"].lower().replace("-", "_") != alg:
                continue

            # Final knowledge: last row's mean_knowledge field
            final_row = rows[-1]
            knowledge_keys = [k for k in final_row if "knowledge" in k.lower() and "mean" in k.lower()]
            if not knowledge_keys:
                knowledge_keys = [k for k in final_row if "knowledge" in k.lower()]
            if not knowledge_keys:
                continue
            final_knowledge = float(final_row[knowledge_keys[0]])

            # Gini: from quiz counts if available
            count_keys = sorted([k for k in final_row if k.startswith("n_") or "count" in k.lower()])
            if count_keys:
                counts = np.array([float(final_row[k]) for k in count_keys])
                gini_val = gini(counts)
            else:
                gini_val = float("nan")

            return final_knowledge, gini_val

        except Exception:
            continue

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/equibandit")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print(f"\nCollecting from: {results_dir}")
    print("=" * 70)

    summary_rows = []
    missing_count = 0

    for K, lam in CONFIGS:
        print(f"\nK={K}, λ={lam}")
        print(f"  {'Algorithm':<15} {'Knowledge':>12}  {'Gini':>10}  {'N':>4}")
        print(f"  {'-'*15} {'-'*12}  {'-'*10}  {'-'*4}")

        for alg in ALGORITHMS:
            knowledges, ginis = [], []
            for seed in SEEDS:
                k_val, g_val = load_seed_result(results_dir, K, lam, seed, alg)
                if k_val is not None:
                    knowledges.append(k_val)
                    if not np.isnan(g_val):
                        ginis.append(g_val)

            n = len(knowledges)
            if n == 0:
                missing_count += 1
                print(f"  {ALG_LABELS[alg]:<15} {'MISSING':>12}  {'MISSING':>10}  {0:>4}")
                continue

            k_mean = np.mean(knowledges)
            k_se   = np.std(knowledges, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            g_mean = np.mean(ginis)   if ginis else float("nan")
            g_se   = np.std(ginis, ddof=1) / np.sqrt(len(ginis)) if len(ginis) > 1 else 0.0

            print(f"  {ALG_LABELS[alg]:<15} {k_mean:.4f}±{k_se:.4f}  {g_mean:.4f}±{g_se:.4f}  {n:>4}")

            summary_rows.append({
                "K": K, "lambda": lam, "algorithm": ALG_LABELS[alg],
                "n_seeds": n,
                "knowledge_mean": round(k_mean, 4), "knowledge_se": round(k_se, 4),
                "gini_mean": round(g_mean, 4) if not np.isnan(g_mean) else "",
                "gini_se":   round(g_se,   4) if not np.isnan(g_mean) else "",
            })

    # Write summary CSV
    out_path = results_dir / "summary.csv"
    if summary_rows:
        fieldnames = ["K", "lambda", "algorithm", "n_seeds",
                      "knowledge_mean", "knowledge_se", "gini_mean", "gini_se"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary written to: {out_path}")

    if missing_count:
        print(f"\nWARNING: {missing_count} (config, alg) pairs are missing results.")
        print("Check that all 60 SLURM array jobs completed: squeue -u $USER -n equibandit")
    else:
        print("\nAll results present. Ready to update paper tables.")

    # Print LaTeX-ready equity table snippet
    print("\n" + "=" * 70)
    print("LaTeX equity table (K=20, λ=0.01):")
    print("-" * 70)
    for row in summary_rows:
        if row["K"] == 20 and row["lambda"] == 0.01:
            print(f"  {row['algorithm']:<15}  k={row['knowledge_mean']}  Gini={row['gini_mean']}")


if __name__ == "__main__":
    main()
