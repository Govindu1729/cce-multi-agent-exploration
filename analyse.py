"""Analyse results from multiple β runs: Pareto frontier and message protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cce.eval.pareto import pareto_frontier, kneedle_knee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse β-sweep results")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing run subdirs")
    parser.add_argument("--output_dir", type=str, default="analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect results from all subdirectories
    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        eval_file = run_dir / "eval_results.json"
        args_file = run_dir / "args.json"
        if not eval_file.exists() or not args_file.exists():
            continue

        with open(eval_file) as f:
            metrics = json.load(f)
        with open(args_file) as f:
            train_args = json.load(f)

        row = {
            "beta_target": train_args.get("beta", 0.25),
            "run_name": run_dir.name,
            **metrics,
        }
        rows.append(row)

    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "results_table.csv", index=False)
    print(f"Loaded {len(df)} runs.")
    print(df.to_string())

    # --- Pareto analysis ---
    if "bandwidth" in df.columns and "coverage" in df.columns:
        bandwidths = df["bandwidth"].values
        coverages = df["coverage"].values

        front_bw, front_cov, front_idx = pareto_frontier(bandwidths, coverages, maximize_perf=True)

        plt.figure(figsize=(8, 6))
        plt.scatter(bandwidths, coverages, alpha=0.6, label="All runs")
        plt.plot(front_bw, front_cov, "o-", color="red", linewidth=2, label="Pareto frontier")

        # Knee detection
        if len(front_bw) >= 3:
            knee_bw, knee_cov, knee_idx = kneedle_knee(front_bw, front_cov, direction="concave")
            plt.scatter([knee_bw], [knee_cov], s=200, color="gold", marker="*",
                        edgecolor="black", zorder=5, label=f"Knee (β={knee_bw:.2f})")

        plt.xlabel("Bandwidth")
        plt.ylabel("Coverage")
        plt.title("Bandwidth-Performance Pareto Frontier")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "pareto_frontier.png", dpi=150)
        plt.close()

        # Save Pareto data
        pareto_df = pd.DataFrame({"bandwidth": front_bw, "coverage": front_cov})
        pareto_df.to_csv(output_dir / "pareto_data.csv", index=False)

    # --- Message entropy vs β ---
    if "entropy" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df["beta_target"], df["entropy"], alpha=0.7)
        plt.xlabel("Target β")
        plt.ylabel("Message Entropy")
        plt.title("Message Entropy vs Communication Budget")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "entropy_vs_beta.png", dpi=150)
        plt.close()

    # --- T95 vs β ---
    if "t95" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df["beta_target"], df["t95"], alpha=0.7)
        plt.xlabel("Target β")
        plt.ylabel("T95 (steps to 95% coverage)")
        plt.title("Exploration Speed vs Communication Budget")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "t95_vs_beta.png", dpi=150)
        plt.close()

    print(f"\nAnalysis saved to {output_dir}")


if __name__ == "__main__":
    main()
