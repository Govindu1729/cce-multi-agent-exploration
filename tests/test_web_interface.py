"""Unit and edge-case tests for the Streamlit dashboard (app.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import app


def test_empty_trajectory():
    """Ensure precomputing metrics on an empty trajectory returns safe defaults."""
    m = app.precompute_trajectory_metrics(())
    assert m["final_coverage"] == 0.0
    assert m["t95_step"] == -1
    assert m["coverage_hist"] == []
    assert m["bandwidth_hist"] == []


def test_single_frame_trajectory():
    """Ensure 1-frame trajectory computes without index or division errors."""
    single_frame = {
        "step": 0,
        "grid": np.zeros((20, 20), dtype=int),
        "visited_global": np.zeros((20, 20), dtype=int),
        "agent_positions": [[0, 0], [19, 19]],
        "messages": [],
        "metrics": {"coverage": 0.05, "bandwidth_used": 0.0},
    }
    m = app.precompute_trajectory_metrics((single_frame,))
    assert len(m["coverage_hist"]) == 1
    fig = app.grid_figure(single_frame, 0)
    assert fig is not None


def test_grid_corner_agents_and_fov():
    """Ensure agent positions at the exact boundary [0,0] and [G-1, G-1] clamp safely."""
    G = 50
    corner_frame = {
        "step": 0,
        "grid": np.ones((G, G), dtype=int),
        "visited_global": np.ones((G, G), dtype=int),
        "agent_positions": [[0, 0], [0, G - 1], [G - 1, 0], [G - 1, G - 1]],
        "messages": [],
        "metrics": {"coverage": 1.0, "bandwidth_used": 0.0},
    }
    fig = app.grid_figure(corner_frame, 0, fov_k=7)
    assert fig is not None
    assert len(fig.layout.shapes) == 4


def test_active_communication_messages():
    """Ensure active communication links render with directional markers."""
    msg_frame = {
        "step": 10,
        "grid": np.zeros((20, 20), dtype=int),
        "visited_global": np.ones((20, 20), dtype=int),
        "agent_positions": [[5, 5], [5, 7], [10, 10], [10, 12]],
        "messages": [
            {"sender": 0, "receiver": 1, "code": 3, "utility": 0.85},
            {"sender": 2, "receiver": 3, "code": 7, "utility": 0.95},
        ],
        "metrics": {"coverage": 0.75, "bandwidth_used": 120.0},
    }
    fig = app.grid_figure(msg_frame, 10, show_comm_range=True, comm_range_manhattan=3)
    assert fig is not None
    # Check that scatter traces include active comm lines
    names = [t.name for t in fig.data if hasattr(t, "name")]
    assert any("Active Comm" in str(n) for n in names)


def test_pareto_frontier_edge_cases():
    """Ensure Pareto frontier handles 1 point, identical bandwidths, and identical values."""
    # 1 point
    bw1, cov1, _ = app.pareto_frontier_calc(np.array([10.0]), np.array([0.5]))
    assert len(bw1) == 1

    # Identical bandwidths: keeps the upper envelope (highest coverage)
    bw_same, cov_same, _ = app.pareto_frontier_calc(
        np.array([100.0, 100.0, 100.0]), np.array([0.5, 0.7, 0.6])
    )
    assert len(bw_same) == 1
    assert cov_same[0] == 0.7

    # Knee detection on identical points
    knee_bw, knee_cov, _ = app.kneedle_knee_calc(
        np.array([50.0, 50.0, 50.0]), np.array([0.8, 0.8, 0.8])
    )
    assert knee_bw == 50.0


def test_corrupted_data_loaders():
    """Ensure corrupted bytes do not crash the app."""
    assert app.load_trajectory_bytes(b"corrupted pickle bytes") == []
    assert app.load_json_bytes(b"{invalid json") == {}
    assert app.load_csv_bytes(b"").empty


def test_real_trajectories_loading():
    """Ensure both real trajectories on disk load and precompute correctly."""
    runs = app.scan_directory_runs()
    assert len(runs["pkl"]) >= 2
    for p in runs["pkl"]:
        traj = app.load_trajectory_from_disk(str(p))
        assert len(traj) == 128
        m = app.precompute_trajectory_metrics(tuple(traj))
        assert m["final_coverage"] > 0.0


def test_get_frame_edge_cases():
    """Ensure get_frame safely handles empty, None, and object-based inputs."""
    f_empty = app.get_frame({}, 0)
    assert f_empty["grid"].shape == (10, 10)
    assert f_empty["agent_positions"] == []

    f_none = app.get_frame(None, 0)
    assert f_none["grid"].shape == (10, 10)

    class CustomObjStep:
        def __init__(self):
            self.step = 3
            self.grid = np.ones((8, 8))
            self.agent_positions = [[2, 2]]
            self.messages = []

    f_obj = app.get_frame(CustomObjStep(), 3)
    assert f_obj["step"] == 3
    assert f_obj["grid"].shape == (8, 8)


def test_synchronized_figures_empty_inputs():
    """Ensure all synchronized plot functions handle empty lists without crashing."""
    assert app.coverage_figure([], 0) is not None
    assert app.bandwidth_figure([], 0) is not None
    assert app.overlap_figure([], 0) is not None


def test_pareto_minimization_and_knee():
    """Ensure Pareto frontier supports minimization (e.g. T95 steps) and finds knees."""
    # Minimize T95 steps: lower is better
    bws = np.array([0.0, 100.0, 250.0])
    t95s = np.array([128.0, 80.0, 50.0])
    fbw, ft95, _ = app.pareto_frontier_calc(bws, t95s, maximize_perf=False)
    assert len(fbw) == 3

    # Knee detection on realistic diminishing returns curve
    cov_curve = np.array([0.35, 0.70, 0.92, 0.95])
    bw_curve = np.array([0.0, 100.0, 250.0, 500.0])
    kbw, kcov, kidx = app.kneedle_knee_calc(bw_curve, cov_curve)
    assert kbw == 250.0
    assert kcov == 0.92


def test_comparison_figures():
    """Ensure Pareto and bar chart comparison figures render on empty and populated tables."""
    # Empty DataFrame
    assert app.pareto_frontier_figure(pd.DataFrame()) is not None
    b1, b2 = app.comparison_bar_charts(pd.DataFrame())
    assert b1 is not None and b2 is not None

    # Populated comparison DataFrame
    sample_df = pd.DataFrame([
        {
            "Run": "Random",
            "Bandwidth (bits)": 0.0,
            "Final Coverage (%)": 48.0,
            "T95 (steps)": 128.0,
            "Comm Efficiency": 0.0,
        },
        {
            "Run": "Frontier",
            "Bandwidth (bits)": 0.0,
            "Final Coverage (%)": 100.0,
            "T95 (steps)": 19.0,
            "Comm Efficiency": 0.0,
        },
        {
            "Run": "CCE (Target)",
            "Bandwidth (bits)": 240.0,
            "Final Coverage (%)": 96.0,
            "T95 (steps)": 28.0,
            "Comm Efficiency": 0.004,
        },
    ])
    fig_pareto = app.pareto_frontier_figure(sample_df)
    assert fig_pareto is not None
    bar_cov, bar_eff = app.comparison_bar_charts(sample_df)
    assert bar_cov is not None and bar_eff is not None
