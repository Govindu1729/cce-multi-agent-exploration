"""Procedural map generation with connectivity guarantees."""

from __future__ import annotations

import numpy as np


def _connected_components(free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iterative BFS connected-component labelling.

    Args:
        free: (H, W) bool array, True = free cell.

    Returns:
        labels: (H, W) int64, -1 for non-free cells, component id otherwise.
        sizes: (n_components,) int64
    """
    H, W = free.shape
    labels = np.full((H, W), -1, dtype=np.int64)
    sizes: list[int] = []
    current_id = 0
    stack: list[tuple[int, int]] = []

    for y in range(H):
        for x in range(W):
            if free[y, x] and labels[y, x] == -1:
                # BFS
                stack.append((y, x))
                labels[y, x] = current_id
                count = 0
                while stack:
                    cy, cx = stack.pop()
                    count += 1
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and free[ny, nx] and labels[ny, nx] == -1:
                            labels[ny, nx] = current_id
                            stack.append((ny, nx))
                sizes.append(count)
                current_id += 1
    return labels, np.array(sizes, dtype=np.int64)


def keep_largest_component(obstacle_map: np.ndarray) -> np.ndarray:
    """Zero out free cells not in the largest connected component.

    Returns a new map where only the largest free-space component remains free;
    other free cells become obstacles. This guarantees that agents cannot be
    spawned into unreachable pockets.
    """
    H, W = obstacle_map.shape
    free = obstacle_map == 0
    labels, sizes = _connected_components(free)
    if sizes.size == 0:
        # Entire map is obstacle — carve a tiny free area
        new_map = np.ones_like(obstacle_map)
        new_map[H // 2, W // 2] = 0
        return new_map
    largest = int(sizes.argmax())
    keep = labels == largest
    new_map = np.where(keep, 0, 1).astype(np.int8)
    return new_map


def generate_map(rng: np.random.Generator, grid_size: int, obstacle_density: float,
                 smooth_iters: int = 0, min_free_frac: float = 0.3) -> np.ndarray:
    """Generate a (grid_size, grid_size) obstacle map with connectivity.

    Args:
        rng: NumPy Generator.
        grid_size: side length.
        obstacle_density: target fraction of obstacle cells before smoothing.
        smooth_iters: number of cellular-automaton smoothing passes (0 = pure random).
        min_free_frac: minimum acceptable fraction of free cells after connectivity carve-out.

    Returns:
        map: (grid_size, grid_size) int8 with 1 = obstacle, 0 = free.
    """
    H = W = grid_size
    for _attempt in range(20):
        # Outer wall of obstacles (optional, but keeps agents safely inside)
        m = np.zeros((H, W), dtype=np.int8)
        m[0, :] = 1; m[-1, :] = 1
        m[:, 0] = 1; m[:, -1] = 1

        # Random obstacles in the interior
        interior = rng.random((H - 2, W - 2)) < obstacle_density
        m[1:-1, 1:-1] = interior.astype(np.int8)

        # Optional smoothing (produces cave-like structures)
        for _ in range(smooth_iters):
            new_m = m.copy()
            # Count 4-neighbourhood obstacles (including self)
            padded = np.pad(m, 1, mode="constant", constant_values=1)
            neigh = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:] + m)
            new_m[1:-1, 1:-1] = (neigh >= 3).astype(np.int8)
            m = new_m

        # Carve to largest connected free component
        m = keep_largest_component(m)
        free_frac = (m == 0).mean()
        if free_frac >= min_free_frac:
            return m
    # Fallback: empty interior
    m = np.zeros((H, W), dtype=np.int8)
    m[0, :] = 1; m[-1, :] = 1; m[:, 0] = 1; m[:, -1] = 1
    return m


def sample_free_positions(rng: np.random.Generator, obstacle_map: np.ndarray, n: int) -> np.ndarray:
    """Sample n distinct free positions uniformly.

    Returns:
        (n, 2) int64 array of (y, x) coordinates.
    """
    free_y, free_x = np.nonzero(obstacle_map == 0)
    if free_y.size < n:
        raise ValueError(f"Not enough free cells ({free_y.size}) to place {n} agents.")
    idx = rng.choice(free_y.size, size=n, replace=False)
    return np.stack([free_y[idx], free_x[idx]], axis=1).astype(np.int64)
