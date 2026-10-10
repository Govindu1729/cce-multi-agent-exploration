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
    """Compute overlap metric: fraction of observed cells already covered.

    Args:
        visited_global: (B, G, G) bool
        visited_local: (B, N, G, G) bool

    Returns:
        mean overlap fraction across envs.
    """
    # For each agent, count cells in its visited_local that were also in visited_global
    # This is a rough proxy for "redundant exploration"
    overlap_counts = (visited_local & visited_global[:, None, :, :]).sum(axis=(1, 2, 3))
    total_observed = visited_local.sum(axis=(1, 2, 3))
    overlap_frac = overlap_counts / (total_observed + 1e-8)
    return float(overlap_frac.mean())
