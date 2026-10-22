"""Episode recorder for visualising multi-agent exploration rollouts.

This module is deliberately dependency-light: it only reads state that the
environment already exposes (maps, agent positions, visited masks, in-range
matrix) and never mutates the environment or the learning code.

The recorder produces a pickled list of ``EpisodeStep`` objects that the
Streamlit dashboard (``app.py``) replays frame by frame.

It is designed to hook into *any* evaluation loop — MAPPO, a frontier
baseline, or a random policy — because it only needs the environment plus the
per-step ``send_flags`` that were executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Distinct colours used by the dashboard, indexed by agent id (0-based).
AGENT_COLOURS = (
    "#ef4444",  # agent 1 - red
    "#3b82f6",  # agent 2 - blue
    "#22c55e",  # agent 3 - green
    "#f59e0b",  # agent 4 - amber
    "#a855f7",  # agent 5 - purple
    "#ec4899",  # agent 6 - pink
    "#14b8a6",  # agent 7 - teal
    "#f97316",  # agent 8 - orange
)


@dataclass
class EpisodeStep:
    """A single recorded timestep of one environment.

    Attributes:
        step: Zero-based timestep index within the episode.
        grid: (G, G) int array — 0 = free, 1 = obstacle.
        visited_global: (G, G) int array — 0 = unvisited, 1..N = first agent id
            that observed the cell.
        agent_positions: list of [y, x] for each agent (env coordinate order).
        messages: list of message dicts sent this step, each with keys
            ``sender``, ``receiver``, ``code``, ``utility``.
        metrics: dict with ``coverage`` (0..1), ``bandwidth_used`` (cumulative
            bits sent so far) and ``lam`` (coverage / bandwidth efficiency).
    """

    step: int
    grid: np.ndarray
    visited_global: np.ndarray
    agent_positions: list[list[int]]
    messages: list[dict[str, Any]]
    metrics: dict[str, float] = field(default_factory=dict)


class EpisodeRecorder:
    """Accumulates ``EpisodeStep`` frames for a single environment.

    Usage::

        rec = EpisodeRecorder(env, env.maps[0])
        obs = env.reset()
        rec.reset()
        for t in range(max_steps):
            ...
            obs, rew, done, info = env.step(move, send, emb)
            rec.record_step(obs, info, send_flags=send, message_indices=msg_idx)
            if done[0]:
                break
        rec.save("runs/viz/trajectory.pkl")
    """

    def __init__(self, env, grid: np.ndarray, batch_index: int = 0, dm_bits: int = 4):
        """Initialise the recorder.

        Args:
            env: A ``VectorGridWorld`` instance (read-only access).
            grid: (G, G) obstacle map for the recorded environment.
            batch_index: Which environment in the batch to record (default 0).
            dm_bits: Message dimension in bits, used for the bandwidth metric.
        """
        self.env = env
        self.grid = np.asarray(grid).copy()
        self.G = self.grid.shape[0]
        self.N = env.N
        self.b = batch_index
        self.dm_bits = dm_bits
        self.trajectory: list[EpisodeStep] = []
        self._bandwidth_used = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the trajectory and per-episode accumulators."""
        self.trajectory = []
        self._bandwidth_used = 0.0

    # ------------------------------------------------------------------
    def _visited_global_map(self) -> np.ndarray:
        """First-visitor map: 0 = unvisited, 1..N = agent id that saw it first."""
        vg = np.zeros((self.G, self.G), dtype=int)
        local = self.env.visited_local[self.b]  # (N, G, G)
        for i in range(self.N):
            first_visit = local[i] & (vg == 0)
            vg[first_visit] = i + 1
        return vg

    def _agent_positions(self) -> list[list[int]]:
        pos = self.env.agent_pos[self.b]  # (N, 2)
        return [[int(y), int(x)] for y, x in pos]

    def _messages(self, send_flags: np.ndarray | None,
                  message_indices: np.ndarray | None) -> list[dict[str, Any]]:
        """Expand send flags + in-range mask into per-link message records."""
        messages: list[dict[str, Any]] = []
        if send_flags is None:
            return messages

        in_range = self.env.in_range[self.b]  # (N, N)
        flags = np.asarray(send_flags)[self.b] if np.asarray(send_flags).ndim == 2 \
            else np.asarray(send_flags)
        for sender in range(self.N):
            if not bool(flags[sender]):
                continue
            for receiver in range(self.N):
                if sender == receiver or not bool(in_range[sender, receiver]):
                    continue
                code = 0
                if message_indices is not None:
                    mi = np.asarray(message_indices)
                    if mi.ndim == 3:
                        code = int(mi[self.b, sender, 0])
                    elif mi.ndim == 2:
                        code = int(mi[sender, 0])
                messages.append({
                    "sender": sender,
                    "receiver": receiver,
                    "code": code,
                    "utility": 1.0,
                })
        return messages

    # ------------------------------------------------------------------
    def record_step(
        self,
        obs: dict[str, np.ndarray] | None,
        info: dict[str, Any],
        send_flags: np.ndarray | None = None,
        message_indices: np.ndarray | None = None,
    ) -> None:
        """Append the current environment state as a frame.

        Args:
            obs: The observation dict returned by ``env.step`` (unused for the
                core layers but kept for API symmetry).
            info: The ``infos`` dict returned by ``env.step``.
            send_flags: (B, N) bool array of send decisions just executed.
            message_indices: (B, N, n_slots) int array of VQ codes, if available.
        """
        coverage = float(np.asarray(info["coverage"]).reshape(-1)[self.b])

        messages = self._messages(send_flags, message_indices)
        self._bandwidth_used += len(messages) * self.dm_bits

        lam = coverage / (self._bandwidth_used + 1e-6)

        self.trajectory.append(EpisodeStep(
            step=len(self.trajectory),
            grid=self.grid.copy(),
            visited_global=self._visited_global_map(),
            agent_positions=self._agent_positions(),
            messages=messages,
            metrics={
                "coverage": coverage,
                "bandwidth_used": float(self._bandwidth_used),
                "lam": float(lam),
            },
        ))

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Pickle the trajectory to ``path`` (creating parent dirs)."""
        import os
        import pickle

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.trajectory, fh)

    @staticmethod
    def load(path: str) -> list[EpisodeStep]:
        """Load a previously saved trajectory."""
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)