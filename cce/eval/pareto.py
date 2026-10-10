"""Pareto frontier analysis and knee detection."""

from __future__ import annotations

import numpy as np


def pareto_frontier(
    bandwidths: np.ndarray,
    performances: np.ndarray,
    maximize_perf: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Pareto frontier for bandwidth vs. performance.

    Args:
        bandwidths: (N,) bandwidth values.
        performances: (N,) performance values (e.g., coverage).
        maximize_perf: if True, we want high performance with low bandwidth.

    Returns:
        front_bw: bandwidths on the Pareto frontier.
        front_perf: performances on the Pareto frontier.
        front_idx: indices of frontier points.
    """
    points = np.stack([bandwidths, performances], axis=1)

    # Sort by bandwidth ascending
    order = np.argsort(bandwidths)
    points_sorted = points[order]

    if maximize_perf:
        # For each bandwidth, keep the max performance (upper envelope)
        front_idx_list = []
        max_perf = -np.inf
        for i, (bw, perf) in enumerate(points_sorted):
            if perf > max_perf:
                max_perf = perf
                front_idx_list.append(i)
        front_idx_sorted = np.array(front_idx_list, dtype=int)
    else:
        # Lower envelope
        front_idx_list = []
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


def kneedle_knee(
    bandwidths: np.ndarray,
    performances: np.ndarray,
    direction: str = "concave",
) -> tuple[float, float, int]:
    """Find the knee point using the Kneedle algorithm.

    The knee is the point of maximum curvature on the Pareto frontier.

    Args:
        bandwidths: (N,) sorted bandwidth values.
        performances: (N,) performance values.
        direction: 'concave' for diminishing returns, 'convex' for increasing.

    Returns:
        knee_bw: bandwidth at the knee.
        knee_perf: performance at the knee.
        knee_idx: index in the input arrays.
    """
    if len(bandwidths) < 3:
        # Not enough points for knee detection
        return float(bandwidths.mean()), float(performances.mean()), 0

    # Normalise to [0, 1]
    bw_min, bw_max = bandwidths.min(), bandwidths.max()
    perf_min, perf_max = performances.min(), performances.max()
    if bw_max == bw_min:
        bw_norm = np.zeros_like(bandwidths)
    else:
        bw_norm = (bandwidths - bw_min) / (bw_max - bw_min)
    if perf_max == perf_min:
        perf_norm = np.zeros_like(performances)
    else:
        perf_norm = (performances - perf_min) / (perf_max - perf_min)

    # Line from first to last point
    x0, y0 = bw_norm[0], perf_norm[0]
    x1, y1 = bw_norm[-1], perf_norm[-1]
    line_dx = x1 - x0
    line_dy = y1 - y0
    line_len = np.sqrt(line_dx**2 + line_dy**2)

    if line_len < 1e-8:
        # All points are the same
        return float(bandwidths.mean()), float(performances.mean()), 0

    # Distance from each point to the line
    # d = |(x - x0) * line_dy - (y - y0) * line_dx| / line_len
    distances = np.abs(
        (bw_norm - x0) * line_dy - (perf_norm - y0) * line_dx
    ) / line_len

    if direction == "concave":
        # Knee is the point farthest above the line (diminishing returns)
        knee_idx = int(np.argmax(distances))
    else:
        knee_idx = int(np.argmax(distances))

    return float(bandwidths[knee_idx]), float(performances[knee_idx]), knee_idx
