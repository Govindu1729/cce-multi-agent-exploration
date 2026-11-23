"""Streamlit dashboard for replaying recorded exploration trajectories and comparing runs.

Run with::

    streamlit run app.py

Features:
- Zero-CLI Trajectory & Results Loader (directory browser + drag-and-drop uploader)
- Interactive Grid Replay with FOV overlays, cell-visit overlap, and Manhattan communication links
- Smooth playback controls (Play/Pause, Step +/- 1, Speed control, Timeline scrubber)
- Synchronized per-frame metrics with 95% T95 reference line
- Multi-run side-by-side comparison and Bandwidth-Performance Pareto frontier analysis
- Full theme-adaptive support (flawless contrast in both Dark and Light modes)
"""

from __future__ import annotations

import io
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Color palette for distinct agent visualization (vibrant in both dark and light modes)
AGENT_COLOURS = [
    "#ef4444",  # Agent 1: Red
    "#3b82f6",  # Agent 2: Blue
    "#10b981",  # Agent 3: Emerald Green
    "#f59e0b",  # Agent 4: Amber
    "#a855f7",  # Agent 5: Purple
    "#ec4899",  # Agent 6: Pink
    "#06b6d4",  # Agent 7: Cyan
    "#f97316",  # Agent 8: Orange
]

APP_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIRS = [
    APP_DIR / "runs" / "viz",
    APP_DIR / "runs",
    Path("runs/viz"),
    Path("runs"),
    Path("."),
]


# ----------------------------------------------------------------------
# Page Config & Adaptive Theme Styling
# ----------------------------------------------------------------------
def apply_page_styling() -> None:
    """Apply Streamlit page configuration and theme-adaptive CSS styling."""
    try:
        st.set_page_config(
            page_title="CCE Multi-Agent Exploration Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    try:
        st.markdown(
            """
            <style>
              .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
              h1 { margin-bottom: 0.1rem; }

              /* Fully theme-adaptive metric boxes (works in both Dark and Light modes) */
              div[data-testid="stMetric"] {
                  background-color: var(--secondary-background-color, rgba(128, 128, 128, 0.08)) !important;
                  border: 1px solid rgba(128, 128, 128, 0.25) !important;
                  border-radius: 10px !important;
                  padding: 12px 14px !important;
                  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
              }
              div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
                  font-size: 1.65rem !important;
                  font-weight: 700 !important;
                  color: var(--text-color, inherit) !important;
              }
              div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
                  font-size: 0.86rem !important;
                  font-weight: 600 !important;
                  color: var(--text-color, inherit) !important;
                  opacity: 0.9 !important;
              }
              div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {
                  font-size: 0.80rem !important;
              }

              /* Control buttons styling */
              div.stButton > button {
                  border-radius: 8px !important;
                  font-weight: 600 !important;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass


# ----------------------------------------------------------------------
# Cached Data Loaders
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_trajectory_bytes(data: bytes) -> list:
    """Unpickle trajectory from raw bytes. Cached by content."""
    try:
        return pickle.load(io.BytesIO(data))
    except Exception as exc:
        st.error(f"Failed to unpickle trajectory data: {exc}")
        return []


@st.cache_data(show_spinner=False)
def load_trajectory_from_disk(path_str: str) -> list:
    """Read and unpickle a trajectory from a file path."""
    p = Path(path_str)
    if not p.exists():
        return []
    return load_trajectory_bytes(p.read_bytes())


@st.cache_data(show_spinner=False)
def load_json_bytes(data: bytes) -> dict:
    """Load JSON evaluation results from bytes."""
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:
        st.error(f"Failed to parse JSON file: {exc}")
        return {}


@st.cache_data(show_spinner=False)
def load_json_from_disk(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        return {}
    return load_json_bytes(p.read_bytes())


@st.cache_data(show_spinner=False)
def load_csv_bytes(data: bytes) -> pd.DataFrame:
    """Load CSV evaluation table from bytes."""
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        st.error(f"Failed to parse CSV file: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_csv_from_disk(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    return load_csv_bytes(p.read_bytes())


def get_frame(record: Any, idx: int = 0) -> dict:
    """Normalize an EpisodeStep dataclass or dict into a standardized dictionary."""
    if isinstance(record, dict):
        grid = record.get("grid")
        grid_arr = np.asarray(grid) if grid is not None else np.zeros((10, 10), dtype=int)
        visited = record.get("visited_global")
        visited_arr = np.asarray(visited) if visited is not None else np.zeros_like(grid_arr)
        return {
            "step": record.get("step", idx),
            "grid": grid_arr,
            "visited_global": visited_arr,
            "agent_positions": record.get("agent_positions", []),
            "messages": record.get("messages", []),
            "metrics": record.get("metrics", {}),
        }
    grid = getattr(record, "grid", None)
    grid_arr = np.asarray(grid) if grid is not None else np.zeros((10, 10), dtype=int)
    visited = getattr(record, "visited_global", None)
    visited_arr = np.asarray(visited) if visited is not None else np.zeros_like(grid_arr)
    return {
        "step": getattr(record, "step", idx),
        "grid": grid_arr,
        "visited_global": visited_arr,
        "agent_positions": getattr(record, "agent_positions", []),
        "messages": getattr(record, "messages", []),
        "metrics": getattr(record, "metrics", {}),
    }


def scan_directory_runs(base_dirs: list[Path] | None = None) -> dict[str, list[Path]]:
    """Scan directory hierarchy for .pkl, .json, and .csv files."""
    if base_dirs is None:
        base_dirs = DEFAULT_RUNS_DIRS

    found: dict[str, list[Path]] = {"pkl": [], "json": [], "csv": []}
    seen = set()

    for b in base_dirs:
        if not b.exists():
            continue
        try:
            for ext in ["*.pkl", "*.json", "*.csv"]:
                for p in b.glob(ext):
                    rp = p.resolve()
                    if rp not in seen and not p.name.startswith("."):
                        seen.add(rp)
                        found[ext[2:]].append(p)
                for p in b.glob(f"*/*/{ext}"):
                    rp = p.resolve()
                    if rp not in seen and not p.name.startswith("."):
                        seen.add(rp)
                        found[ext[2:]].append(p)
        except Exception:
            continue

    for k in found:
        found[k].sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)

    return found


# ----------------------------------------------------------------------
# Metrics & Overlap Analysis Helpers
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def precompute_trajectory_metrics(trajectory_tuple: tuple) -> dict[str, Any]:
    """Compute per-frame metrics and cumulative overlap masks across an entire trajectory."""
    if not trajectory_tuple:
        return {
            "coverage_hist": [],
            "bandwidth_hist": [],
            "messages_per_step": [],
            "cumulative_messages": [],
            "overlap_hist": [],
            "t95_step": -1,
            "final_coverage": 0.0,
            "final_overlap": 0.0,
            "final_bandwidth": 0.0,
            "comm_efficiency": 0.0,
        }

    first_frame = get_frame(trajectory_tuple[0], 0)
    grid = first_frame["grid"]
    G = grid.shape[0]
    n_agents = len(first_frame["agent_positions"])

    coverage_hist = []
    bandwidth_hist = []
    messages_per_step = []
    cumulative_messages = []
    overlap_hist = []

    # Track per-agent cumulative visit masks to compute exact overlap at each frame
    fov_k = 5
    half_k = fov_k // 2
    agent_visited = np.zeros((n_agents, G, G), dtype=bool)

    cum_msgs = 0
    t95_step = -1

    for t, rec in enumerate(trajectory_tuple):
        f = get_frame(rec, t)
        cov = float(f.get("metrics", {}).get("coverage", 0.0))
        bw = float(f.get("metrics", {}).get("bandwidth_used", 0.0))
        msgs = f.get("messages", [])

        coverage_hist.append(cov)
        bandwidth_hist.append(bw)
        messages_per_step.append(len(msgs))
        cum_msgs += len(msgs)
        cumulative_messages.append(cum_msgs)

        if t95_step == -1 and cov >= 0.95:
            t95_step = t

        positions = f["agent_positions"]
        for a_idx, pos in enumerate(positions):
            if a_idx < n_agents and len(pos) >= 2:
                ay, ax = int(pos[0]), int(pos[1])
                y0, y1 = max(0, ay - half_k), min(G, ay + half_k + 1)
                x0, x1 = max(0, ax - half_k), min(G, ax + half_k + 1)
                agent_visited[a_idx, y0:y1, x0:x1] = True

        visit_counts = agent_visited.sum(axis=0)
        tot_vis = (visit_counts > 0).sum()
        ovlp_cells = (visit_counts >= 2).sum()
        ovlp_frac = float(ovlp_cells / max(tot_vis, 1)) if tot_vis > 0 else 0.0
        overlap_hist.append(ovlp_frac)

    final_cov = coverage_hist[-1] if coverage_hist else 0.0
    final_bw = bandwidth_hist[-1] if bandwidth_hist else 0.0
    final_overlap = overlap_hist[-1] if overlap_hist else 0.0

    cov0 = coverage_hist[0] if coverage_hist else 0.0
    delta_cov = max(0.0, final_cov - cov0)
    comm_eff = delta_cov / final_bw if final_bw > 0 else 0.0

    return {
        "coverage_hist": coverage_hist,
        "bandwidth_hist": bandwidth_hist,
        "messages_per_step": messages_per_step,
        "cumulative_messages": cumulative_messages,
        "overlap_hist": overlap_hist,
        "t95_step": t95_step,
        "final_coverage": final_cov,
        "final_overlap": final_overlap,
        "final_bandwidth": final_bw,
        "comm_efficiency": comm_eff,
    }


def pareto_frontier_calc(
    bandwidths: np.ndarray,
    performances: np.ndarray,
    maximize_perf: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Pareto upper frontier without external dependency."""
    if len(bandwidths) == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    points = np.stack([bandwidths, performances], axis=1)
    if maximize_perf:
        # Sort bandwidth ascending, and tie-break performance descending so upper points come first
        order = np.lexsort((-performances, bandwidths))
    else:
        order = np.lexsort((performances, bandwidths))
    points_sorted = points[order]

    front_idx_list = []
    if maximize_perf:
        max_perf = -np.inf
        for i, (bw, perf) in enumerate(points_sorted):
            if perf > max_perf:
                max_perf = perf
                front_idx_list.append(i)
    else:
        min_perf = np.inf
        for i, (bw, perf) in enumerate(points_sorted):
            if perf < min_perf:
                min_perf = perf
                front_idx_list.append(i)

    front_idx_sorted = np.array(front_idx_list, dtype=int)
    front_bw = points_sorted[front_idx_sorted, 0]
    front_perf = points_sorted[front_idx_sorted, 1]
    front_idx = order[front_idx_sorted]
    return front_bw, front_perf, front_idx


def kneedle_knee_calc(
    bandwidths: np.ndarray,
    performances: np.ndarray,
) -> tuple[float, float, int]:
    """Compute knee point of diminishing returns on the frontier curve."""
    if len(bandwidths) < 3:
        return float(bandwidths.mean()), float(performances.mean()), 0

    bw_min, bw_max = bandwidths.min(), bandwidths.max()
    perf_min, perf_max = performances.min(), performances.max()

    bw_norm = np.zeros_like(bandwidths) if bw_max == bw_min else (bandwidths - bw_min) / (bw_max - bw_min)
    perf_norm = np.zeros_like(performances) if perf_max == perf_min else (performances - perf_min) / (perf_max - perf_min)

    x0, y0 = bw_norm[0], perf_norm[0]
    x1, y1 = bw_norm[-1], perf_norm[-1]
    line_dx = x1 - x0
    line_dy = y1 - y0
    line_len = np.sqrt(line_dx**2 + line_dy**2)

    if line_len < 1e-8:
        return float(bandwidths.mean()), float(performances.mean()), 0

    distances = np.abs((bw_norm - x0) * line_dy - (perf_norm - y0) * line_dx) / line_len
    knee_idx = int(np.argmax(distances))
    return float(bandwidths[knee_idx]), float(performances[knee_idx]), knee_idx


# ----------------------------------------------------------------------
# Plotly Replay Figure Builders (Theme-Adaptive & Transparent)
# ----------------------------------------------------------------------
def grid_figure(
    frame: dict,
    step_idx: int,
    cumulative_agent_visited: np.ndarray | None = None,
    show_fov: bool = True,
    fov_k: int = 5,
    show_comm_range: bool = True,
    comm_range_manhattan: int = 3,
    highlight_overlap: bool = True,
) -> go.Figure:
    """Construct Plotly grid figure with obstacle map, visited cells, overlap, FOV, and comm links."""
    grid = np.asarray(frame["grid"])
    visited = np.asarray(frame["visited_global"])
    positions = frame["agent_positions"]
    messages = frame["messages"]
    G = grid.shape[0]
    n_agents = len(positions)

    fig = go.Figure()

    # Layer 1 — Terrain: Free unexplored cells (soft silver-gray #e2e8f0) and Obstacles (slate black #0f172a)
    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale=[[0, "#e2e8f0"], [1, "#0f172a"]],
        zmin=0, zmax=1,
        showscale=False,
        hoverinfo="text",
        text=[["Obstacle" if grid[r, c] == 1 else "Unexplored Free Cell" for c in range(G)] for r in range(G)],
    ))

    # Layer 2 — Visited cells colored by the first agent that discovered each cell
    vmask = visited.astype(float).copy()
    vmask[vmask == 0] = np.nan
    colourscale = [[0, "rgba(0,0,0,0)"]]
    for i in range(1, n_agents + 1):
        colourscale.append([i / max(n_agents, 1), AGENT_COLOURS[(i - 1) % len(AGENT_COLOURS)]])

    fig.add_trace(go.Heatmap(
        z=vmask,
        colorscale=colourscale,
        zmin=1, zmax=max(n_agents, 1),
        showscale=False,
        opacity=0.72,
        hoverinfo="skip",
    ))

    # Layer 3 — Overlap highlighting (cells observed by >= 2 agents)
    if highlight_overlap and cumulative_agent_visited is not None:
        overlap_mask = (cumulative_agent_visited.sum(axis=0) >= 2)
        if overlap_mask.any():
            oy, ox = np.where(overlap_mask)
            fig.add_trace(go.Scatter(
                x=ox, y=oy,
                mode="markers",
                marker=dict(symbol="square", size=9, color="#f59e0b",
                            line=dict(width=1, color="#78350f")),
                name="Overlap (>=2)",
                hovertemplate="Overlap Cell (>=2 agents)<br>Coordinates: (x=%{x}, y=%{y})<extra></extra>",
            ))

    # Layer 4 — k x k Local FOV Overlays around each agent
    if show_fov:
        half_k = fov_k // 2
        for i, pos in enumerate(positions):
            if len(pos) < 2:
                continue
            ay, ax = pos[0], pos[1]
            c = AGENT_COLOURS[i % len(AGENT_COLOURS)]
            x_min = max(-0.5, ax - half_k - 0.5)
            x_max = min(G - 0.5, ax + half_k + 0.5)
            y_min = max(-0.5, ay - half_k - 0.5)
            y_max = min(G - 0.5, ay + half_k + 0.5)

            fig.add_shape(
                type="rect",
                x0=x_min, y0=y_min, x1=x_max, y1=y_max,
                line=dict(color=c, width=1.5, dash="dot"),
                fillcolor=c,
                opacity=0.15,
                layer="above",
            )

    # Layer 5 — Potential Communication Links (Manhattan distance <= 3)
    if show_comm_range and n_agents > 1:
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if i < len(positions) and j < len(positions):
                    y1, x1 = positions[i]
                    y2, x2 = positions[j]
                    dist = abs(y1 - y2) + abs(x1 - x2)
                    if dist <= comm_range_manhattan:
                        fig.add_trace(go.Scatter(
                            x=[x1, x2], y=[y1, y2],
                            mode="lines",
                            line=dict(color="rgba(148, 163, 184, 0.7)", width=1.5, dash="dot"),
                            hoverinfo="text",
                            text=f"In Range: Agent {i+1} <-> Agent {j+1} (Manhattan d={dist})",
                            showlegend=False,
                        ))

    # Layer 6 — Active Communication Transmissions (this timestep)
    for msg in messages:
        s, r = msg.get("sender", -1), msg.get("receiver", -1)
        if 0 <= s < n_agents and 0 <= r < n_agents and s != r:
            sy, sx = positions[s]
            ry, rx = positions[r]
            code = msg.get("code", 0)
            util = msg.get("utility", 1.0)
            sc = AGENT_COLOURS[s % len(AGENT_COLOURS)]

            fig.add_trace(go.Scatter(
                x=[sx, rx], y=[sy, ry],
                mode="lines+markers",
                line=dict(color=sc, width=3.5),
                marker=dict(symbol="arrow-bar-up", size=11, color=sc, angleref="previous"),
                hoverinfo="text",
                text=f"Transmission: Agent {s+1} -> Agent {r+1}<br>Code: {code} | Utility: {util:.2f}",
                name=f"Active Comm {s+1}->{r+1}",
                showlegend=False,
            ))

    # Layer 7 — Agent Positions (Markers + Distinct Colors)
    for i, pos in enumerate(positions):
        ay, ax = pos[0], pos[1]
        c = AGENT_COLOURS[i % len(AGENT_COLOURS)]
        fig.add_trace(go.Scatter(
            x=[ax], y=[ay],
            mode="markers+text",
            marker=dict(symbol="star", size=22, color=c, line=dict(width=2, color="#ffffff")),
            text=[f"A{i+1}"],
            textposition="top center",
            textfont=dict(color=c, size=11, family="sans-serif"),
            name=f"Agent {i+1}",
            hovertemplate=f"<b>Agent {i+1}</b><br>Position: (y={ay}, x={ax})<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"Episode Replay — Timestep {step_idx} | Active Messages: {len(messages)}",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(
            range=[-0.5, G - 0.5],
            dtick=1,
            constrain="domain",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.22)",
            zeroline=False,
            showline=True,
            linecolor="rgba(128, 128, 128, 0.4)",
        ),
        yaxis=dict(
            range=[G - 0.5, -0.5],  # Inverted matrix coordinate convention
            dtick=1,
            scaleanchor="x",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.22)",
            zeroline=False,
            showline=True,
            linecolor="rgba(128, 128, 128, 0.4)",
        ),
        height=620,
        margin=dict(l=25, r=25, t=50, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def coverage_figure(coverage_hist: list[float], step_idx: int) -> go.Figure:
    fig = go.Figure()
    steps = list(range(len(coverage_hist)))

    fig.add_trace(go.Scatter(
        x=steps, y=coverage_hist,
        mode="lines",
        line=dict(color="#3b82f6", width=2.8),
        name="Coverage Cov(t)",
        hovertemplate="Step %{x}: Cov=%{y:.1%}<extra></extra>",
    ))

    # 95% Reference Target line
    fig.add_trace(go.Scatter(
        x=[0, max(len(coverage_hist) - 1, 1)],
        y=[0.95, 0.95],
        mode="lines",
        line=dict(color="#10b981", width=2.0, dash="dash"),
        name="95% Target (T95)",
        hoverinfo="skip",
    ))

    # Current step marker
    if not coverage_hist:
        current_val = 0.0
    else:
        current_val = coverage_hist[step_idx] if step_idx < len(coverage_hist) else coverage_hist[-1]
    fig.add_trace(go.Scatter(
        x=[step_idx], y=[current_val],
        mode="markers",
        marker=dict(size=12, color="#ef4444", line=dict(width=2, color="#ffffff")),
        name="Current Step",
        hovertemplate=f"Current Step: {step_idx}<br>Coverage: {current_val:.1%}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="Exploration Coverage: Cov(t)", font=dict(size=14)),
        xaxis=dict(title="Timestep", showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        yaxis=dict(title="Coverage", tickformat=".0%", range=[0, 1.05], showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        height=270,
        margin=dict(l=35, r=20, t=40, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    )
    return fig


def bandwidth_figure(bandwidth_hist: list[float], step_idx: int) -> go.Figure:
    fig = go.Figure()
    steps = list(range(len(bandwidth_hist)))

    fig.add_trace(go.Scatter(
        x=steps, y=bandwidth_hist,
        mode="lines",
        line=dict(color="#f59e0b", width=2.4),
        name="Cumulative Bandwidth",
        hovertemplate="Step %{x}: %{y:.0f} bits<extra></extra>",
    ))

    if not bandwidth_hist:
        current_val = 0.0
    else:
        current_val = bandwidth_hist[step_idx] if step_idx < len(bandwidth_hist) else bandwidth_hist[-1]
    fig.add_trace(go.Scatter(
        x=[step_idx], y=[current_val],
        mode="markers",
        marker=dict(size=11, color="#ef4444", line=dict(width=1.5, color="#ffffff")),
        name="Current",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="Cumulative Bandwidth (bits)", font=dict(size=14)),
        xaxis=dict(title="Timestep", showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        yaxis=dict(title="Bits Sent", showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        height=250,
        margin=dict(l=35, r=20, t=40, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def overlap_figure(overlap_hist: list[float], step_idx: int) -> go.Figure:
    fig = go.Figure()
    steps = list(range(len(overlap_hist)))

    fig.add_trace(go.Scatter(
        x=steps, y=overlap_hist,
        mode="lines",
        line=dict(color="#8b5cf6", width=2.4),
        name="Overlap Fraction",
        hovertemplate="Step %{x}: %{y:.1%}<extra></extra>",
    ))

    if not overlap_hist:
        current_val = 0.0
    else:
        current_val = overlap_hist[step_idx] if step_idx < len(overlap_hist) else overlap_hist[-1]
    fig.add_trace(go.Scatter(
        x=[step_idx], y=[current_val],
        mode="markers",
        marker=dict(size=11, color="#ef4444", line=dict(width=1.5, color="#ffffff")),
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="Redundant Exploration Overlap Fraction", font=dict(size=14)),
        xaxis=dict(title="Timestep", showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        yaxis=dict(title="Overlap Fraction", tickformat=".1%", range=[0, 1.05], showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        height=250,
        margin=dict(l=35, r=20, t=40, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ----------------------------------------------------------------------
# Pareto & Comparison Plots (Theme-Adaptive & Transparent)
# ----------------------------------------------------------------------
def pareto_frontier_figure(comparison_df: pd.DataFrame) -> go.Figure:
    """Plot Bandwidth vs Coverage Pareto frontier with knee point highlighting."""
    fig = go.Figure()

    if comparison_df.empty or "Bandwidth (bits)" not in comparison_df.columns:
        return fig

    valid_df = comparison_df.dropna(subset=["Bandwidth (bits)", "Final Coverage (%)"]).copy()
    if valid_df.empty:
        return fig

    bws = valid_df["Bandwidth (bits)"].values.astype(float)
    covs = (valid_df["Final Coverage (%)"] / 100.0).values.astype(float)
    names = valid_df["Run"].values

    # Scatter of all evaluated runs
    fig.add_trace(go.Scatter(
        x=bws, y=covs,
        mode="markers",
        marker=dict(size=13, color="#38bdf8", line=dict(width=1.5, color="#0284c7")),
        text=names,
        hovertemplate="<b>%{text}</b><br>Bandwidth: %{x:.1f} bits<br>Coverage: %{y:.1%}<extra></extra>",
        name="Evaluated Runs",
    ))

    # Compute Pareto Frontier
    front_bw, front_cov, front_idx = pareto_frontier_calc(bws, covs, maximize_perf=True)
    if len(front_bw) > 0:
        fig.add_trace(go.Scatter(
            x=front_bw, y=front_cov,
            mode="lines+markers",
            line=dict(color="#ef4444", width=2.8, dash="solid"),
            marker=dict(size=9, color="#ef4444"),
            name="Pareto Frontier",
        ))

        # Knee detection if >= 3 frontier points
        if len(front_bw) >= 3:
            knee_bw, knee_cov, _ = kneedle_knee_calc(front_bw, front_cov)
            fig.add_trace(go.Scatter(
                x=[knee_bw], y=[knee_cov],
                mode="markers",
                marker=dict(symbol="star", size=20, color="#fbbf24", line=dict(width=1.5, color="#78350f")),
                name=f"Optimal Knee (BW={knee_bw:.0f})",
                hovertemplate=f"<b>Knee Optimal Point</b><br>Bandwidth: {knee_bw:.1f}<br>Coverage: {knee_cov:.1%}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text="Bandwidth–Coverage Pareto Frontier", font=dict(size=16)),
        xaxis=dict(title="Cumulative Bandwidth (bits)", showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        yaxis=dict(title="Final Coverage", tickformat=".0%", range=[0, 1.05], showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        height=480,
        margin=dict(l=40, r=30, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def comparison_bar_charts(comparison_df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    """Generate comparative bar figures for Coverage and Efficiency."""
    fig_cov = go.Figure()
    fig_eff = go.Figure()

    if comparison_df.empty:
        return fig_cov, fig_eff

    runs = comparison_df["Run"].tolist()

    # Bar chart 1: Coverage and Success
    if "Final Coverage (%)" in comparison_df.columns:
        fig_cov.add_trace(go.Bar(
            x=runs, y=comparison_df["Final Coverage (%)"],
            name="Final Coverage (%)",
            marker_color="#3b82f6",
        ))
    if "T95 (steps)" in comparison_df.columns:
        fig_cov.add_trace(go.Bar(
            x=runs, y=comparison_df["T95 (steps)"],
            name="T95 Steps",
            marker_color="#10b981",
        ))
    fig_cov.update_layout(
        title="Exploration Performance & Speed",
        barmode="group",
        height=320,
        margin=dict(l=30, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
    )

    # Bar chart 2: Bandwidth and Communication Efficiency
    if "Bandwidth (bits)" in comparison_df.columns:
        fig_eff.add_trace(go.Bar(
            x=runs, y=comparison_df["Bandwidth (bits)"],
            name="Bandwidth (bits)",
            marker_color="#f59e0b",
        ))
    if "Comm Efficiency" in comparison_df.columns:
        fig_eff.add_trace(go.Bar(
            x=runs, y=comparison_df["Comm Efficiency"] * 1000,
            name="Comm Eff (x1000)",
            marker_color="#8b5cf6",
        ))
    fig_eff.update_layout(
        title="Communication Footprint & Efficiency",
        barmode="group",
        height=320,
        margin=dict(l=30, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.20)"),
    )

    return fig_cov, fig_eff


# ----------------------------------------------------------------------
# Application Main
# ----------------------------------------------------------------------
def main() -> None:
    apply_page_styling()
    st.title("🤖 CCE Multi-Agent Exploration — Analytics & Replay")
    st.caption("Zero-CLI evaluation inspector: frame-by-frame trajectory replay, communication graphs, and Pareto frontier comparison.")

    # Initialize playback state
    if "step_idx" not in st.session_state:
        st.session_state.step_idx = 0
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "selected_run_name" not in st.session_state:
        st.session_state.selected_run_name = None

    # Discover runs on disk
    scanned_files = scan_directory_runs()

    # ------------------------------------------------------------------
    # Sidebar: File Uploader & Run Discovery
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("📂 Data & Trajectories")

        # Upload files directly via browser
        uploaded_files = st.file_uploader(
            "Upload runs (.pkl, .json, .csv)",
            type=["pkl", "json", "csv"],
            accept_multiple_files=True,
            help="Upload trajectory.pkl or evaluation summary JSON/CSV files.",
        )

        run_registry: dict[str, dict[str, Any]] = {}

        if uploaded_files:
            for uf in uploaded_files:
                ext = uf.name.split(".")[-1].lower()
                key = f"[Upload] {uf.name}"
                run_registry[key] = {"type": ext, "source": uf, "name": uf.name}

        for p in scanned_files["pkl"]:
            rel = str(p).replace("\\", "/")
            key = f"[Local] {rel}"
            run_registry[key] = {"type": "pkl", "source": p, "name": p.stem}

        for p in scanned_files["json"]:
            rel = str(p).replace("\\", "/")
            key = f"[Local JSON] {rel}"
            run_registry[key] = {"type": "json", "source": p, "name": p.stem}

        for p in scanned_files["csv"]:
            rel = str(p).replace("\\", "/")
            key = f"[Local CSV] {rel}"
            run_registry[key] = {"type": "csv", "source": p, "name": p.stem}

        trajectory_keys = [k for k, v in run_registry.items() if v["type"] == "pkl"]

        if not run_registry:
            st.warning("No runs found in `runs/viz/` or `runs/`.")
            st.info("Upload a `.pkl` trajectory above or generate one with:\n\n"
                    "`python scripts/evaluate.py --baseline frontier --record --record_output runs/viz/trajectory.pkl`")
            st.stop()

        st.subheader("🎯 Active Trajectory")
        selected_key = st.selectbox(
            "Select trajectory for replay:",
            options=trajectory_keys if trajectory_keys else list(run_registry.keys()),
            index=0,
            help="Switch between loaded runs without restarting backend scripts.",
        )

        st.divider()
        st.subheader("⚙️ Replay Display Options")
        show_fov = st.checkbox("Show Agent FOV Boxes", value=True)
        fov_size = st.slider("FOV Size (k x k)", min_value=3, max_value=9, value=5, step=2) if show_fov else 5
        show_comm_range = st.checkbox("Show Comm Range (Manhattan <= 3)", value=True)
        highlight_overlap = st.checkbox("Highlight Visited Overlap (>= 2)", value=True)

    # ------------------------------------------------------------------
    # Data Loading for Active Replay
    # ------------------------------------------------------------------
    active_trajectory: list = []
    if selected_key and run_registry[selected_key]["type"] == "pkl":
        entry = run_registry[selected_key]
        src = entry["source"]
        if hasattr(src, "getvalue"):
            active_trajectory = load_trajectory_bytes(src.getvalue())
        else:
            active_trajectory = load_trajectory_from_disk(str(src))

    traj_metrics = precompute_trajectory_metrics(tuple(active_trajectory)) if active_trajectory else {}
    n_frames = len(active_trajectory) if active_trajectory else 0

    if n_frames > 0:
        if st.session_state.step_idx >= n_frames:
            st.session_state.step_idx = n_frames - 1
    else:
        st.session_state.step_idx = 0

    # ------------------------------------------------------------------
    # Main View Tabs
    # ------------------------------------------------------------------
    tab1, tab2 = st.tabs(["🎥 Tab 1: Episode Replay", "📊 Tab 2: Multi-Run & Pareto Comparison"])

    # ==================================================================
    # TAB 1: Episode Replay
    # ==================================================================
    with tab1:
        if not active_trajectory:
            st.info("Select a `.pkl` trajectory from the sidebar to inspect frame-by-frame replay.")
        else:
            current_step = st.session_state.step_idx
            frame = get_frame(active_trajectory[current_step], current_step)
            frame_metrics = frame.get("metrics", {})

            grid = frame["grid"]
            G = grid.shape[0]
            n_agents = len(frame["agent_positions"])
            cum_agent_vis = np.zeros((n_agents, G, G), dtype=bool)
            half_k = fov_size // 2

            for t in range(current_step + 1):
                f_t = get_frame(active_trajectory[t], t)
                for a_i, p in enumerate(f_t["agent_positions"]):
                    if a_i < n_agents and len(p) >= 2:
                        ay, ax = p[0], p[1]
                        y0, y1 = max(0, ay - half_k), min(G, ay + half_k + 1)
                        x0, x1 = max(0, ax - half_k), min(G, ax + half_k + 1)
                        cum_agent_vis[a_i, y0:y1, x0:x1] = True

            # Synchronized Per-Frame Metrics Bar
            m_cov = float(frame_metrics.get("coverage", 0.0))
            m_bw = float(frame_metrics.get("bandwidth_used", 0.0))
            m_ovlp = traj_metrics["overlap_hist"][current_step] if traj_metrics.get("overlap_hist") else 0.0
            step_msgs = len(frame["messages"])
            cum_msgs = traj_metrics["cumulative_messages"][current_step] if traj_metrics.get("cumulative_messages") else 0
            cov0 = traj_metrics["coverage_hist"][0] if traj_metrics.get("coverage_hist") else 0.0
            eff_now = (m_cov - cov0) / m_bw if m_bw > 0 else 0.0

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Coverage Cov(t)", f"{m_cov:.1%}", delta=f"{m_cov - cov0:+.1%}")
            m2.metric("Overlap (>=2)", f"{m_ovlp:.1%}")
            m3.metric("Bandwidth Used", f"{m_bw:.0f} bits")
            m4.metric("Step Messages", f"{step_msgs}")
            m5.metric("Total Messages", f"{cum_msgs}")
            m6.metric("Comm Efficiency", f"{eff_now:.4f}" if m_bw > 0 else "—")

            # Playback Controls Bar
            c_first, c_prev, c_play, c_next, c_last, c_speed = st.columns([1, 1, 1.4, 1, 1, 2.5])

            with c_first:
                if st.button("⏮ First", use_container_width=True):
                    st.session_state.step_idx = 0
                    st.session_state.is_playing = False
                    st.rerun()

            with c_prev:
                if st.button("◀ Prev", use_container_width=True):
                    st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
                    st.session_state.is_playing = False
                    st.rerun()

            with c_play:
                play_label = "⏸ Pause" if st.session_state.is_playing else "▶ Play"
                if st.button(play_label, use_container_width=True, type="primary" if st.session_state.is_playing else "secondary"):
                    st.session_state.is_playing = not st.session_state.is_playing
                    st.rerun()

            with c_next:
                if st.button("Next ▶", use_container_width=True):
                    st.session_state.step_idx = min(n_frames - 1, st.session_state.step_idx + 1)
                    st.session_state.is_playing = False
                    st.rerun()

            with c_last:
                if st.button("Last ⏭", use_container_width=True):
                    st.session_state.step_idx = n_frames - 1
                    st.session_state.is_playing = False
                    st.rerun()

            with c_speed:
                speed = st.select_slider(
                    "Speed",
                    options=["0.5x", "1.0x", "2.0x", "4.0x"],
                    value="1.0x",
                    label_visibility="collapsed",
                )

            slider_val = st.slider(
                "Timeline Scrubber (Timestep)",
                min_value=0,
                max_value=n_frames - 1,
                value=st.session_state.step_idx,
                key="timeline_slider",
            )
            if slider_val != st.session_state.step_idx:
                st.session_state.step_idx = slider_val
                st.session_state.is_playing = False
                st.rerun()

            left_col, right_col = st.columns([1.7, 1.3], gap="large")

            with left_col:
                fig_grid = grid_figure(
                    frame=frame,
                    step_idx=current_step,
                    cumulative_agent_visited=cum_agent_vis,
                    show_fov=show_fov,
                    fov_k=fov_size,
                    show_comm_range=show_comm_range,
                    highlight_overlap=highlight_overlap,
                )
                st.plotly_chart(fig_grid, use_container_width=True)

                st.caption(
                    "⬛ Obstacle · ⬜ Free Unexplored · Coloured = Visited (Agent 1..N) · "
                    "🟨 Golden Dot = Visited Overlap (>= 2 Agents) · ⭐ Agent Position · ➖ Comm Link"
                )

            with right_col:
                st.plotly_chart(
                    coverage_figure(traj_metrics["coverage_hist"], current_step),
                    use_container_width=True,
                )

                if any(b > 0 for b in traj_metrics["bandwidth_hist"]):
                    st.plotly_chart(
                        bandwidth_figure(traj_metrics["bandwidth_hist"], current_step),
                        use_container_width=True,
                    )
                else:
                    st.plotly_chart(
                        overlap_figure(traj_metrics["overlap_hist"], current_step),
                        use_container_width=True,
                    )

            st.divider()
            msg_col1, msg_col2 = st.columns([1, 1])

            with msg_col1:
                st.subheader(f"📨 Active Transmissions at Step {current_step}")
                if frame["messages"]:
                    df_msg = pd.DataFrame(frame["messages"])
                    df_msg["sender"] = df_msg["sender"].apply(lambda s: f"Agent {s + 1}")
                    df_msg["receiver"] = df_msg["receiver"].apply(lambda r: f"Agent {r + 1}")
                    st.dataframe(df_msg, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No messages transmitted at timestep {current_step}.")

            with msg_col2:
                with st.expander("📜 Full Episode Message Log"):
                    all_messages = []
                    for t, r in enumerate(active_trajectory):
                        f_r = get_frame(r, t)
                        for m in f_r.get("messages", []):
                            all_messages.append({
                                "step": t,
                                "sender": f"Agent {m['sender'] + 1}",
                                "receiver": f"Agent {m['receiver'] + 1}",
                                "code": m.get("code", 0),
                                "utility": round(float(m.get("utility", 1.0)), 2),
                            })
                    if all_messages:
                        st.dataframe(pd.DataFrame(all_messages), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No communication messages occurred throughout this entire episode.")

            # Auto-advance for Play mode
            if st.session_state.is_playing:
                delay_map = {"0.5x": 0.25, "1.0x": 0.12, "2.0x": 0.05, "4.0x": 0.01}
                time.sleep(delay_map.get(speed, 0.12))
                if st.session_state.step_idx < n_frames - 1:
                    st.session_state.step_idx += 1
                else:
                    st.session_state.step_idx = 0
                st.rerun()

    # ==================================================================
    # TAB 2: Multi-Run & Pareto Comparison
    # ==================================================================
    with tab2:
        st.subheader("📊 Multi-Run Benchmark & Pareto Frontier")
        st.caption("Compare multiple evaluation trajectories, summary tables, or β-sweeps without recomputing backend metrics.")

        all_available_runs = list(run_registry.keys())
        default_selected = all_available_runs[:min(len(all_available_runs), 4)]

        selected_comparison_keys = st.multiselect(
            "Select 2 or more runs to compare:",
            options=all_available_runs,
            default=default_selected,
            help="Choose pre-recorded .pkl trajectories or evaluation result files (.json, .csv).",
        )

        if len(selected_comparison_keys) < 1:
            st.warning("Please select at least one run to inspect comparison metrics.")
        else:
            comparison_rows = []

            for k in selected_comparison_keys:
                item = run_registry[k]
                itype = item["type"]
                src = item["source"]
                name = item["name"]

                if itype == "pkl":
                    traj = load_trajectory_bytes(src.getvalue()) if hasattr(src, "getvalue") else load_trajectory_from_disk(str(src))
                    m = precompute_trajectory_metrics(tuple(traj))
                    comparison_rows.append({
                        "Run": name,
                        "Type": "Trajectory (.pkl)",
                        "Final Coverage (%)": round(m["final_coverage"] * 100, 1),
                        "T95 (steps)": m["t95_step"] if m["t95_step"] != -1 else len(traj),
                        "Overlap (%)": round(m["final_overlap"] * 100, 1),
                        "Success (>=95%)": "Yes" if m["final_coverage"] >= 0.95 else "No",
                        "Bandwidth (bits)": round(m["final_bandwidth"], 0),
                        "Comm Efficiency": round(m["comm_efficiency"], 5),
                    })

                elif itype == "json":
                    data = load_json_bytes(src.getvalue()) if hasattr(src, "getvalue") else load_json_from_disk(str(src))
                    if data:
                        cov = data.get("coverage", 0.0)
                        bw = data.get("bandwidth_bits", data.get("bandwidth", 0.0))
                        comparison_rows.append({
                            "Run": name,
                            "Type": "Summary (.json)",
                            "Final Coverage (%)": round(cov * 100, 1) if cov is not None else 0.0,
                            "T95 (steps)": round(data.get("t95", 0.0), 1) if data.get("t95") is not None else 0.0,
                            "Overlap (%)": round(data.get("overlap", 0.0) * 100, 1) if data.get("overlap") is not None else 0.0,
                            "Success (>=95%)": "Yes" if data.get("success_rate", 0.0) >= 0.95 else "No",
                            "Bandwidth (bits)": round(bw, 0) if bw is not None else 0.0,
                            "Comm Efficiency": round(data.get("comm_efficiency", 0.0), 5) if data.get("comm_efficiency") is not None else 0.0,
                        })

                elif itype == "csv":
                    df_csv = load_csv_bytes(src.getvalue()) if hasattr(src, "getvalue") else load_csv_from_disk(str(src))
                    if not df_csv.empty:
                        for _, row in df_csv.iterrows():
                            r_name = row.get("run_name", f"{name}_{row.name}")
                            cov = row.get("coverage", 0.0)
                            bw = row.get("bandwidth_bits", row.get("bandwidth", 0.0))
                            comparison_rows.append({
                                "Run": str(r_name),
                                "Type": "Table (.csv)",
                                "Final Coverage (%)": round(float(cov) * 100, 1) if pd.notna(cov) else 0.0,
                                "T95 (steps)": round(float(row.get("t95", 0.0)), 1) if pd.notna(row.get("t95")) else 0.0,
                                "Overlap (%)": round(float(row.get("overlap", 0.0)) * 100, 1) if pd.notna(row.get("overlap")) else 0.0,
                                "Success (>=95%)": "Yes" if float(row.get("success_rate", 0.0)) >= 0.95 else "No",
                                "Bandwidth (bits)": round(float(bw), 0) if pd.notna(bw) else 0.0,
                                "Comm Efficiency": round(float(row.get("comm_efficiency", 0.0)), 5) if pd.notna(row.get("comm_efficiency")) else 0.0,
                            })

            comp_df = pd.DataFrame(comparison_rows)

            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            st.divider()
            pareto_col, bar_col = st.columns([1.5, 1.2], gap="large")

            with pareto_col:
                st.plotly_chart(pareto_frontier_figure(comp_df), use_container_width=True)
                st.caption(
                    "🔴 Red line: Empirical Pareto frontier (upper envelope of coverage vs bandwidth). "
                    "⭐ Gold star: Optimal knee point balancing communication cost with exploration returns."
                )

            with bar_col:
                fig_cov_bar, fig_eff_bar = comparison_bar_charts(comp_df)
                st.plotly_chart(fig_cov_bar, use_container_width=True)
                st.plotly_chart(fig_eff_bar, use_container_width=True)


if __name__ == "__main__":
    main()
