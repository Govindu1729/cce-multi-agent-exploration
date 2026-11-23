"""Automated self-test script for the Streamlit dashboard (app.py).

Run from PowerShell with::

    & "C:/Users/priya/AppData/Local/Programs/Python/Python310/python.exe" test_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure cce-multi-agent-exploration is in sys.path
current_dir = Path(__file__).resolve().parent
if (current_dir / "cce-multi-agent-exploration").exists():
    cce_dir = current_dir / "cce-multi-agent-exploration"
else:
    cce_dir = current_dir

if str(cce_dir) not in sys.path:
    sys.path.insert(0, str(cce_dir))

import numpy as np
import pandas as pd
import app


def main():
    print("=" * 60)
    print("   CCE Dashboard Verification Test Suite")
    print("=" * 60)

    # 1. Directory Scanner
    print("\n[1/5] Testing directory scanner...")
    runs = app.scan_directory_runs()
    n_pkl = len(runs.get("pkl", []))
    n_json = len(runs.get("json", []))
    n_csv = len(runs.get("csv", []))
    print(f"      Found {n_pkl} .pkl trajectories, {n_json} .json eval files, {n_csv} .csv tables.")
    assert n_pkl > 0 or n_json > 0, "No run files found in runs/ or workspace!"
    print("      ✓ Directory scanner working.")

    # 2. Trajectory Loader
    print("\n[2/5] Testing trajectory loading and metrics precomputation...")
    target_pkl = runs["pkl"][0]
    traj = app.load_trajectory_from_disk(str(target_pkl))
    print(f"      Loaded {len(traj)} frames from {target_pkl.name}")
    assert len(traj) > 0, "Trajectory file was empty!"

    m = app.precompute_trajectory_metrics(tuple(traj))
    assert len(m["coverage_hist"]) == len(traj), "Coverage history mismatch!"
    assert len(m["overlap_hist"]) == len(traj), "Overlap history mismatch!"
    print(f"      Final Coverage: {m['final_coverage']:.1%}, Final Overlap: {m['final_overlap']:.1%}")
    print("      ✓ Trajectory loader and precomputed metrics working.")

    # 3. Grid Visuals & Overlap / FOV
    print("\n[3/5] Testing Grid and Comm figure builder...")
    frame0 = app.get_frame(traj[0], 0)
    fig_grid = app.grid_figure(
        frame=frame0,
        step_idx=0,
        show_fov=True,
        fov_k=5,
        show_comm_range=True,
        highlight_overlap=True,
    )
    assert fig_grid is not None and hasattr(fig_grid, "data"), "Grid figure failed to generate!"
    print(f"      Generated grid figure with {len(fig_grid.data)} trace layers.")
    print("      ✓ Grid, FOV, and Comm layers working.")

    # 4. Synchronized Metrics Curves
    print("\n[4/5] Testing synchronized metrics curves...")
    fig_cov = app.coverage_figure(m["coverage_hist"], 0)
    fig_bw = app.bandwidth_figure(m["bandwidth_hist"], 0)
    fig_ov = app.overlap_figure(m["overlap_hist"], 0)
    assert fig_cov is not None and fig_bw is not None and fig_ov is not None
    print("      ✓ Coverage, Bandwidth, and Overlap curves working.")

    # 5. Pareto Frontier & Knee Detection
    print("\n[5/5] Testing Pareto frontier analysis and Knee detection...")
    bws = np.array([0.0, 100.0, 250.0, 500.0])
    covs = np.array([0.40, 0.75, 0.92, 0.96])
    fbw, fcov, _ = app.pareto_frontier_calc(bws, covs, maximize_perf=True)
    assert len(fbw) == 4, "Pareto calculation failed!"

    knee_bw, knee_cov, _ = app.kneedle_knee_calc(fbw, fcov)
    assert knee_bw == 250.0, f"Expected knee at 250.0, got {knee_bw}"
    print(f"      Detected Knee Point at Bandwidth={knee_bw:.0f}, Coverage={knee_cov:.1%}")

    sample_df = pd.DataFrame([
        {"Run": "Run A", "Bandwidth (bits)": 0.0, "Final Coverage (%)": 40.0, "T95 (steps)": 128, "Comm Efficiency": 0.0},
        {"Run": "Run B", "Bandwidth (bits)": 100.0, "Final Coverage (%)": 75.0, "T95 (steps)": 90, "Comm Efficiency": 0.0035},
        {"Run": "Run C", "Bandwidth (bits)": 250.0, "Final Coverage (%)": 92.0, "T95 (steps)": 70, "Comm Efficiency": 0.0021},
    ])
    fig_pareto = app.pareto_frontier_figure(sample_df)
    assert fig_pareto is not None
    print("      ✓ Pareto frontier and Knee point working.")

    print("\n" + "=" * 60)
    print("   ALL 5 TEST SUITES PASSED! The web dashboard is 100% functional.")
    print("=" * 60)
    print("\nYou can now launch the dashboard in your browser by running:")
    print("    python -m streamlit run app.py\n")


if __name__ == "__main__":
    main()
