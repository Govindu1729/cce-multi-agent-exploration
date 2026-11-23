"""Classical (non-learning) baselines for the multi-agent exploration task.

These provide reference points for the MAPPO agent. They only read state the
environment already exposes; they never import or modify the learning stack
(``cce/algo/``) or the communication stack (``cce/comm/``).

Coordinate convention follows ``cce/envs/grid_world.py``:
    Position = (y, x), y increases downward.
    Action 0 = up (dy=-1), 1 = down (dy=+1), 2 = left (dx=-1), 3 = right (dx=+1).
"""

from __future__ import annotations

import numpy as np
from collections import deque

# Action ids (must match cce/envs/grid_world.ACTION_DELTAS).
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_DY = {UP: -1, DOWN: +1, LEFT: 0, RIGHT: 0}
_DX = {UP: 0, DOWN: 0, LEFT: -1, RIGHT: +1}


def _bfs_shortest_action(grid_obstacles, visited_mask, agent_pos, target_cells):
    """Finds shortest obstacle-free path from agent_pos (r, c) to any cell in target_cells."""
    H, W = visited_mask.shape
    sr, sc = int(agent_pos[0]), int(agent_pos[1])
    target_set = set(map(tuple, target_cells))
    if (sr, sc) in target_set:
        target_set.discard((sr, sc))
    if not target_set:
        return np.random.randint(0, 4)

    # Standard GridWorld action deltas: 0: up (-1,0), 1: down (+1,0), 2: left (0,-1), 3: right (0,+1)
    deltas = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]
    queue = deque([(sr, sc, -1)])
    seen = {(sr, sc)}

    while queue:
        r, c, first_act = queue.popleft()
        if (r, c) in target_set and first_act != -1:
            return first_act
        # Randomize neighbor order slightly to break multi-agent symmetry
        for dr, dc, act in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen:
                if grid_obstacles is None or grid_obstacles[nr, nc] == 0:
                    seen.add((nr, nc))
                    queue.append((nr, nc, act if first_act == -1 else first_act))
    return np.random.randint(0, 4)


def _frontier_cells(seen: np.ndarray) -> list[tuple[int, int]]:
    """Cells adjacent (4-conn) to a seen cell but not themselves seen.

    Args:
        seen: (G, G) bool — cells this agent has personally observed.

    Returns:
        List of (y, x) frontier candidates.
    """
    G = seen.shape[0]
    frontier: list[tuple[int, int]] = []
    ys, xs = np.where(~seen)
    for y, x in zip(ys.tolist(), xs.tolist()):
        # adjacency to any seen cell
        if (y > 0 and seen[y - 1, x]) or (y < G - 1 and seen[y + 1, x]) or \
           (x > 0 and seen[y, x - 1]) or (x < G - 1 and seen[y, x + 1]):
            frontier.append((y, x))
    return frontier


def _step_towards(pos: tuple[int, int], target: tuple[int, int]) -> int:
    """Greedy one-step action reducing Manhattan distance to ``target``."""
    y, x = pos
    ty, tx = target
    dy, dx = ty - y, tx - x
    if abs(dy) >= abs(dx) and dy != 0:
        return DOWN if dy > 0 else UP
    if dx != 0:
        return RIGHT if dx > 0 else LEFT
    if dy != 0:
        return DOWN if dy > 0 else UP
    return UP  # already on target


class FrontierBaseline:
    """Frontier-based exploration with a dispersion heuristic.

    Each agent:
      1. Computes frontier cells from the cells *it* has personally seen.
      2. Scores each frontier by Manhattan distance plus a dispersion penalty
         that grows as a frontier sits closer to a teammate.
      3. Steps greedily toward the best-scoring frontier.
    """

    def __init__(self, env, dispersion_weight: float = 1.0):
        self.env = env
        self.G = env.G
        self.N = env.N
        self.dispersion_weight = dispersion_weight

    def act(self, batch_index: int = 0) -> np.ndarray:
        """Return a (1, N) int array of movement actions using global BFS frontier pursuit."""
        env = self.env
        b = batch_index
        positions = [(int(y), int(x)) for y, x in env.agent_pos[b]]
        actions = np.zeros((1, self.N), dtype=np.int64)

        # Use GLOBAL visited map (union of all agents) instead of per-agent local map
        # This prevents agents from getting stuck when their local FOV has no frontier
        if hasattr(env, 'visited_global'):
            global_visited = env.visited_global[b] if env.visited_global.ndim >= 3 else env.visited_global
        else:
            # Fallback: union of per-agent visited masks
            global_visited = env.visited_local[b].any(axis=0)

        # Compute global frontier cells (cells adjacent to visited but not yet visited)
        global_frontiers = _frontier_cells(global_visited)

        if not global_frontiers:
            # No frontiers anywhere - all cells visited or completely blocked
            actions[0, :] = UP
            return actions

        # Get obstacle map if available (VectorGridWorld stores obstacles in env.maps)
        if hasattr(env, 'maps'):
            grid_obstacles = env.maps[b] if env.maps.ndim >= 3 else env.maps
        elif hasattr(env, 'grid'):
            grid_obstacles = env.grid[b] if env.grid.ndim >= 3 else env.grid
        else:
            grid_obstacles = None

        for i in range(self.N):
            y, x = positions[i]

            # Use BFS to find shortest path to nearest frontier
            actions[0, i] = _bfs_shortest_action(
                grid_obstacles,
                global_visited,
                (y, x),
                global_frontiers
            )

        return actions


class RandomBaseline:
    """Uniformly random movement (no communication)."""

    def __init__(self, env, seed: int | None = None):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def act(self, batch_index: int = 0) -> np.ndarray:
        return self.rng.integers(0, self.env.num_actions, size=(1, self.env.N))


def make_baseline(name: str, env, seed: int | None = None, dispersion_weight: float = 1.0):
    """Factory for baselines by name."""
    name = name.lower()
    if name == "frontier":
        return FrontierBaseline(env, dispersion_weight=dispersion_weight)
    if name == "random":
        return RandomBaseline(env, seed=seed)
    raise ValueError(f"Unknown baseline: {name!r} (expected 'frontier' or 'random')")