"""Evaluation metrics for multi-agent exploration."""

from __future__ import annotations

import numpy as np


def compute_episode_metrics(
    coverages: np.ndarray,
    t95_values: np.ndarray,
    comm_costs: np.ndarray,
    rewards: np.ndarray,
    max_steps: int,
) -> dict[str, float]:
    """Compute aggregate metrics over a batch of episodes.

    Args:
        coverages: (B,) final coverage per episode.
        t95_values: (B,) timestep of first reaching 95% coverage, -1 if never.
        comm_costs: (B,) average comm cost per episode.
        rewards: (B,) total reward per episode.
        max_steps: episode length.

    Returns:
        dict of scalar metrics.
    """
    n = coverages.shape[0]
    success = (coverages >= 0.95).mean()
    coverage_mean = coverages.mean()
    coverage_std = coverages.std()

    valid_t95 = t95_values[t95_values >= 0]
    t95_mean = valid_t95.mean() if len(valid_t95) > 0 else max_steps
    t95_std = valid_t95.std() if len(valid_t95) > 0 else 0.0

    avg_comm = comm_costs.mean()
    avg_reward = rewards.mean()

    # Exploration rate = coverage / time
    exploration_rate = coverage_mean / max_steps

    # Communication efficiency = coverage / bandwidth
    # bandwidth ~ avg_comm * T
    bandwidth = avg_comm * max_steps
    comm_efficiency = coverage_mean / (bandwidth + 1e-8)

    return {
        "coverage": float(coverage_mean),
        "coverage_std": float(coverage_std),
        "t95": float(t95_mean),
        "t95_std": float(t95_std),
        "success_rate": float(success),
        "avg_comm_cost": float(avg_comm),
        "bandwidth": float(bandwidth),
        "avg_reward": float(avg_reward),
        "exploration_rate": float(exploration_rate),
        "comm_efficiency": float(comm_efficiency),
    }


def compute_overlap(
    visited_global: np.ndarray,
    visited_local: np.ndarray,
) -> float:
    """Compute overlap: fraction of visited cells seen by >= 2 agents.

    Redundant exploration happens when two or more agents observe the same cell.
    This is measured from per-agent visit counts, NOT from the relationship
    between the per-agent masks and the global union mask (that intersection is
    always the full visited set, which is why the previous definition always
    returned 1.0).

    Args:
        visited_local: (B, N, G, G) or (N, G, G) bool — per-agent observed masks.
        visited_global: (B, G, G) or (G, G) bool — union mask. Accepted for API
            compatibility; the counts are derived from ``visited_local``.

    Returns:
        Mean overlap fraction across the batch, in [0, 1]. 0 means no cell was
        observed by more than one agent; 1 means every visited cell was observed
        by at least two agents.
    """
    vl = np.asarray(visited_local, dtype=bool)

    # Normalise to (B, N, G, G) so the per-cell agent count can be summed.
    if vl.ndim == 3:
        vl = vl[None]
    if vl.ndim != 4:
        raise ValueError(f"visited_local must be (B, N, G, G) or (N, G, G); got {vl.shape}")

    # Per-cell number of agents that have observed it: (B, G, G).
    visit_counts = vl.sum(axis=1)

    visited_cells = visit_counts > 0            # cells seen by at least one agent
    overlap_cells = visit_counts >= 2           # cells seen by two or more agents

    total_visited = visited_cells.sum(axis=(1, 2))          # (B,)
    overlapped = overlap_cells.sum(axis=(1, 2))             # (B,)

    frac = overlapped / np.maximum(total_visited, 1)        # (B,)
    # Environments with no visited cells contribute 0 overlap.
    frac = np.where(total_visited > 0, frac, 0.0)
    return float(frac.mean())
