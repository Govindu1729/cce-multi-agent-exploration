"""Streamlit dashboard for replaying recorded exploration trajectories.

Run with::

    streamlit run app.py

Then either upload a ``trajectory.pkl`` produced by::

    python scripts/evaluate.py --baseline frontier --record \\
        --record_output runs/viz/trajectory.pkl

or click "Load default" if ``runs/viz/trajectory.pkl`` exists.

The dashboard is purely a *viewer*: it reads a pickled list of
``cce.eval.recorder.EpisodeStep`` frames and never touches the training stack.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Colour palette shared with cce/eval/recorder.py (kept local to avoid importing
# torch-dependent modules into the dashboard process).
AGENT_COLOURS = [
    "#ef4444", "#3b82f6", "#22c55e", "#f59e0b",
    "#a855f7", "#ec4899", "#14b8a6", "#f97316",
]
DEFAULT_TRAJECTORY = Path("runs/viz/trajectory.pkl")


st.set_page_config(
    page_title="CCE Multi-Agent Exploration",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; }
      h1 { margin-bottom: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_trajectory(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def get_frame(record, idx: int) -> dict:
    """Normalise a frame (dataclass or dict) into a plain dict."""
    if hasattr(record, "__dict__") and not isinstance(record, dict):
        return {
            "step": getattr(record, "step", idx),
            "grid": record.grid,
            "visited_global": record.visited_global,
            "agent_positions": record.agent_positions,
            "messages": record.messages,
            "metrics": record.metrics,
        }
    return record


# ----------------------------------------------------------------------
# Figure builders
# ----------------------------------------------------------------------
def grid_figure(frame: dict, step_idx: int) -> go.Figure:
    grid = np.asarray(frame["grid"])
    visited = np.asarray(frame["visited_global"])
    positions = frame["agent_positions"]
    messages = frame["messages"]
    G = grid.shape[0]
    n_agents = len(positions)

    fig = go.Figure()

    # Layer 1 — obstacle / free terrain
    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale=[[0, "#f1f5f9"], [1, "#1e293b"]],
        zmin=0, zmax=1,
        showscale=False, hoverinfo="skip",
    ))

    # Layer 2 — visited cells coloured by first-visiting agent
    vmask = visited.astype(float).copy()
    vmask[vmask == 0] = np.nan
    colourscale = [[0, "rgba(0,0,0,0)"]]
    for i in range(1, n_agents + 1):
        colourscale.append([i / max(n_agents, 1), AGENT_COLOURS[(i - 1) % len(AGENT_COLOURS)]])
    fig.add_trace(go.Heatmap(
        z=vmask,
        colorscale=colourscale,
        zmin=1, zmax=max(n_agents, 1),
        showscale=False, opacity=0.75, hoverinfo="skip",
    ))

    # Layer 3 — agent markers
    for i, (y, x) in enumerate(positions):
        colour = AGENT_COLOURS[i % len(AGENT_COLOURS)]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers",
            marker=dict(symbol="star", size=18, color=colour,
                        line=dict(width=1.5, color="white")),
            name=f"Agent {i + 1}",
            hovertemplate=f"Agent {i + 1}<br>(y={y}, x={x})<extra></extra>",
        ))

    # Layer 4 — communication links this step
    for msg in messages:
        s, r = msg["sender"], msg["receiver"]
        if s < n_agents and r < n_agents:
            sy, sx = positions[s]
            ry, rx = positions[r]
            fig.add_trace(go.Scatter(
                x=[sx, rx], y=[sy, ry],
                mode="lines",
                line=dict(color=AGENT_COLOURS[s % len(AGENT_COLOURS)],
                          width=2.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))

    fig.update_layout(
        title=dict(text=f"Timestep {step_idx}", x=0.5, font=dict(size=18)),
        xaxis=dict(range=[-0.5, G - 0.5], dtick=1, constrain="domain",
                   showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(range=[G - 0.5, -0.5], dtick=1, scaleanchor="x",
                   showgrid=True, gridcolor="#e2e8f0"),
        height=620,
        margin=dict(l=30, r=30, t=50, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def coverage_figure(history: list[float], step_idx: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(history))), y=history,
        mode="lines", line=dict(color="#3b82f6", width=3), name="Coverage",
    ))
    fig.add_trace(go.Scatter(
        x=[0, max(len(history) - 1, 1)], y=[0.95, 0.95],
        mode="lines", line=dict(color="#22c55e", width=1.5, dash="dot"),
        name="95% target", hoverinfo="skip",
    ))
    y_now = history[step_idx] if step_idx < len(history) else history[-1]
    fig.add_trace(go.Scatter(
        x=[step_idx], y=[y_now], mode="markers",
        marker=dict(size=11, color="#ef4444"), name="Current", showlegend=False,
    ))
    fig.update_layout(
        title=dict(text="Coverage over time", font=dict(size=15)),
        xaxis_title="Timestep",
        yaxis=dict(title="Coverage", tickformat=".0%", range=[0, 1.05]),
        height=300, margin=dict(l=40, r=20, t=45, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def bandwidth_figure(history: list[float], step_idx: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(history))), y=history,
        mode="lines", line=dict(color="#f59e0b", width=2), name="Bandwidth",
    ))
    y_now = history[step_idx] if step_idx < len(history) else history[-1]
    fig.add_trace(go.Scatter(
        x=[step_idx], y=[y_now], mode="markers",
        marker=dict(size=10, color="#ef4444"), showlegend=False,
    ))
    fig.update_layout(
        title=dict(text="Cumulative bandwidth (bits)", font=dict(size=15)),
        xaxis_title="Timestep", yaxis_title="bits",
        height=240, margin=dict(l=40, r=20, t=45, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    st.title("🤖 CCE Multi-Agent Exploration — Trajectory Viewer")
    st.caption("Replay a recorded episode: coverage, visited cells, agent paths and communication.")

    with st.sidebar:
        st.header("📁 Trajectory")

        uploaded = st.file_uploader("Upload trajectory.pkl", type=["pkl"])
        use_default = False
        if DEFAULT_TRAJECTORY.exists():
            st.success(f"Found default: `{DEFAULT_TRAJECTORY}`")
            use_default = st.button("Load default", use_container_width=True)
        else:
            st.info("No default trajectory found. Record one with:\n\n"
                    "`python scripts/evaluate.py --baseline frontier "
                    "--record --record_output runs/viz/trajectory.pkl`")

        if uploaded is None and not use_default:
            st.stop()

        try:
            if uploaded is not None:
                trajectory = pickle.load(uploaded)
            else:
                trajectory = load_trajectory(str(DEFAULT_TRAJECTORY))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load trajectory: {exc}")
            st.stop()

        if not trajectory:
            st.error("Trajectory is empty.")
            st.stop()

        n = len(trajectory)
        st.success(f"Loaded {n} frames")
        step_idx = st.slider("Timestep", 0, n - 1, 0)

        frame = get_frame(trajectory[step_idx], step_idx)
        metrics = frame.get("metrics", {})

        st.divider()
        st.subheader("📊 Current frame")
        st.metric("Coverage", f"{metrics.get('coverage', 0.0):.1%}")
        st.metric("Bandwidth (bits)", f"{metrics.get('bandwidth_used', 0.0):.0f}")
        if "lam" in metrics:
            st.metric("λ (coverage/bit)", f"{metrics['lam']:.4f}")

        if frame["messages"]:
            st.subheader("📨 Messages")
            df = pd.DataFrame(frame["messages"])
            df["sender"] = df["sender"].apply(lambda i: f"Agent {i + 1}")
            df["receiver"] = df["receiver"].apply(lambda i: f"Agent {i + 1}")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No messages this step")

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.plotly_chart(grid_figure(frame, step_idx), use_container_width=True)
        st.caption("⬛ obstacle · ⬜ unknown · coloured = visited (by first agent) · "
                   "⭐ agent · ➖ communication link")

    with right:
        coverage_hist = [get_frame(r, i).get("metrics", {}).get("coverage", 0.0)
                         for i, r in enumerate(trajectory)]
        st.plotly_chart(coverage_figure(coverage_hist, step_idx), use_container_width=True)

        st.subheader("Progress")
        st.progress((step_idx + 1) / len(trajectory))
        st.caption(f"Frame {step_idx + 1} of {len(trajectory)}")

        bw_hist = [get_frame(r, i).get("metrics", {}).get("bandwidth_used", 0.0)
                   for i, r in enumerate(trajectory)]
        if any(b > 0 for b in bw_hist):
            st.plotly_chart(bandwidth_figure(bw_hist, step_idx), use_container_width=True)

    with st.expander("📜 Full message log"):
        rows = []
        for i, r in enumerate(trajectory):
            f = get_frame(r, i)
            for msg in f.get("messages", []):
                rows.append({
                    "step": i,
                    "sender": f"Agent {msg['sender'] + 1}",
                    "receiver": f"Agent {msg['receiver'] + 1}",
                    "code": msg.get("code", ""),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No messages were sent in this episode.")


if __name__ == "__main__":
    main()