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

# Action ids (must match cce/envs/grid_world.ACTION_DELTAS).
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_DY = {UP: -1, DOWN: +1, LEFT: 0, RIGHT: 0}
_DX = {UP: 0, DOWN: 0, LEFT: -1, RIGHT: +1}


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
        """Return a (1, N) int array of movement actions."""
        env = self.env
        b = batch_index
        positions = [(int(y), int(x)) for y, x in env.agent_pos[b]]
        actions = np.zeros((1, self.N), dtype=np.int64)

        for i in range(self.N):
            seen = env.visited_local[b, i]
            frontiers = _frontier_cells(seen)
            if not frontiers:
                actions[0, i] = UP
                continue

            y, x = positions[i]
            best, best_score = None, np.inf
            for (fy, fx) in frontiers:
                dist = abs(fy - y) + abs(fx - x)
                penalty = 0.0
                for j, (oy, ox) in enumerate(positions):
                    if j == i:
                        continue
                    d = abs(oy - fy) + abs(ox - fx)
                    if d > 0:
                        penalty += 1.0 / d
                score = dist + self.dispersion_weight * penalty
                if score < best_score:
                    best_score, best = score, (fy, fx)

            actions[0, i] = _step_towards((y, x), best)

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