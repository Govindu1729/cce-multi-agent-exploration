"""Vectorised grid-world environment for communication-constrained MARL.

Single-process, NumPy-vectorised implementation that runs B environments in
parallel with no IPC overhead. Episodes have a fixed length `max_steps`; the
success bonus and metrics are recorded at episode boundaries.

Coordinate convention:
    Position = (y, x) with y increasing downward.
    Action 0 = up    (dy = -1)
    Action 1 = down  (dy = +1)
    Action 2 = left  (dx = -1)
    Action 3 = right (dx = +1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cce.envs.mapgen import generate_map, sample_free_positions

ACTION_DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)  # (4, 2)
ACTION_NAMES = ("up", "down", "left", "right")


@dataclass
class GridWorldConfig:
    # World geometry
    grid_size: int = 20
    num_agents: int = 4
    fov_size: int = 5                     # k in the paper (odd recommended)
    obstacle_density: float = 0.2
    smooth_iters: int = 0                 # cellular-automaton smoothing

    # Episode
    max_steps: int = 128

    # Communication
    comm_range: int = 3                   # Manhattan distance threshold
    max_message_dim: int = 32             # d_m^max in the paper

    # Observation
    obs_neighbours: bool = True           # include neighbour relative positions
    obs_self_pos: bool = True             # include agent's own normalised position
    obs_message_embed: bool = True        # include received message embeddings
    silence_embed_dim: int = 0            # if >0, add a learned "silence" embedding

    # Reward shaping
    w_coverage: float = 1.0
    w_overlap: float = 0.1
    w_time: float = 0.01
    w_comm: float = 0.05
    success_bonus: float = 2.0
    success_threshold: float = 0.95
    wall_bump_penalty: float = 0.0        # small penalty for bumping wall/agent

    # Misc
    seed: int = 0
    message_embed_dim: int = 8            # dim of each VQ code embedding (d_code)


class VectorGridWorld:
    """Batched grid-world environment.

    Usage:
        cfg = GridWorldConfig(grid_size=20, num_agents=4)
        env = VectorGridWorld(cfg, num_envs=64)
        obs = env.reset()                 # dict of numpy arrays
        for step in range(cfg.max_steps):
            actions = ...                 # (B, N) int in {0,1,2,3}
            send_flags = ...              # (B, N) bool
            messages = ...                # (B, N, d_m) int in [0, K) or -1
            obs, rewards, dones, infos = env.step(actions, send_flags, messages)
    """

    def __init__(self, cfg: GridWorldConfig, num_envs: int = 1):
        self.cfg = cfg
        self.B = num_envs
        self.N = cfg.num_agents
        self.G = cfg.grid_size
        self.k = cfg.fov_size
        self.r = (self.k - 1) // 2
        assert self.k % 2 == 1, "fov_size must be odd"

        self.rng = np.random.default_rng(cfg.seed)

        # Precompute FOV offsets (k x k window around agent)
        ys, xs = np.meshgrid(np.arange(-self.r, self.r + 1),
                             np.arange(-self.r, self.r + 1), indexing="ij")
        self._fov_offsets = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.int64)  # (k^2, 2)

        # Precompute batch/agent indices for vectorised scatter (reused every step)
        self._b_idx = np.arange(self.B)[:, None].repeat(self.N, axis=1)  # (B, N)
        self._n_idx = np.arange(self.N)[None, :].repeat(self.B, axis=0)  # (B, N)

        # Pairwise agent indices for distance matrix
        self._agent_i, self._agent_j = np.meshgrid(np.arange(self.N), np.arange(self.N), indexing="ij")

        # State arrays (allocated on reset)
        self.maps: np.ndarray = np.zeros((self.B, self.G, self.G), dtype=np.int8)
        self.agent_pos: np.ndarray = np.zeros((self.B, self.N, 2), dtype=np.int64)
        self.visited_local: np.ndarray = np.zeros((self.B, self.N, self.G, self.G), dtype=bool)
        self.visited_global: np.ndarray = np.zeros((self.B, self.G, self.G), dtype=bool)
        self.n_traversable: np.ndarray = np.zeros(self.B, dtype=np.int64)
        self.step_count: np.ndarray = np.zeros(self.B, dtype=np.int64)
        self.t95: np.ndarray = np.full(self.B, -1, dtype=np.int64)

        # Per-step arrays (allocated on step)
        self.observed_now: np.ndarray = np.zeros((self.B, self.G, self.G), dtype=bool)
        self.moved_mask: np.ndarray = np.ones((self.B, self.N), dtype=bool)

        # Comm matrix
        self.in_range: np.ndarray = np.zeros((self.B, self.N, self.N), dtype=bool)

        self._current_obs: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Observation space spec (useful for building networks)
    # ------------------------------------------------------------------
    @property
    def obs_spec(self) -> dict[str, tuple[int, ...]]:
        """Shape of each obs component (without batch dim)."""
        spec: dict[str, tuple[int, ...]] = {
            "local_map": (self.k, self.k),
            "visited_self": (self.k, self.k),
        }
        if self.cfg.obs_self_pos:
            spec["self_pos"] = (2,)
        if self.cfg.obs_neighbours:
            # [in_range, rel_dy/R, rel_dx/R, sent_flag] per other agent
            spec["neighbour_feats"] = (self.N - 1, 4)
        if self.cfg.obs_message_embed:
            # mean-pooled received message embedding + sender count
            spec["message_embed"] = (self.cfg.message_embed_dim + 1,)
        return spec

    @property
    def num_actions(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> dict[str, np.ndarray]:
        """Reset all B environments. Returns initial observation dict."""
        self.step_count[:] = 0
        self.t95[:] = -1

        # Generate maps
        for b in range(self.B):
            self.maps[b] = generate_map(
                self.rng, self.G, self.cfg.obstacle_density, self.cfg.smooth_iters
            )
        self.n_traversable[:] = (self.maps == 0).sum(axis=(1, 2))

        # Place agents on distinct free cells
        for b in range(self.B):
            self.agent_pos[b] = sample_free_positions(self.rng, self.maps[b], self.N)

        # Clear visited
        self.visited_local[:] = False
        self.visited_global[:] = False

        # Initial FOV application — mark starting cells as visited
        self.observed_now[:] = False
        self.moved_mask[:] = True
        self._apply_fov()

        # Compute in-range mask
        self._update_in_range()

        # Build initial observation
        self._current_obs = self._build_observation(
            received_embeddings=np.zeros(
                (self.B, self.N, self.cfg.message_embed_dim), dtype=np.float32
            ),
            received_send_flags=np.zeros((self.B, self.N, self.N), dtype=bool),
        )
        return self._current_obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self,
        actions_move: np.ndarray,
        send_flags: np.ndarray,
        message_embeddings: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
        """Advance one step.

        Args:
            actions_move: (B, N) int in {0,1,2,3}
            send_flags:   (B, N) bool — which agents transmit this step
            message_embeddings: (B, N, d_code) float — VQ embedding of each agent's
                message. Agents with send_flags=False are ignored.

        Returns:
            obs: dict of (B, N, ...) arrays
            rewards: (B, N) float (shared across agents for the same env)
            dones: (B,) bool — True if episode terminated (fixed length here)
            infos: dict with auxiliary info (coverage, bandwidth, t95, etc.)
        """
        assert actions_move.shape == (self.B, self.N)
        assert send_flags.shape == (self.B, self.N)

        prev_visited_global = self.visited_global.copy()

        # ----- Movement -----
        deltas = ACTION_DELTAS[actions_move]                       # (B, N, 2)
        proposed = self.agent_pos + deltas                         # (B, N, 2)
        py = np.clip(proposed[..., 0], 0, self.G - 1)
        px = np.clip(proposed[..., 1], 0, self.G - 1)

        # Collision: wall or obstacle → stay in place
        is_obs = self.maps[self._b_idx, py, px] == 1              # (B, N)
        bumped = is_obs.copy()

        # Agent-agent collisions: if two agents propose the same cell, keep prior positions
        # Simple approach: if any pair collides, neither moves.
        # We compute occupancy and reject duplicates.
        if self.N > 1:
            # Encode proposed positions as unique keys per env
            keys = py * self.G + px                                # (B, N)
            # Sort within each batch and find duplicates
            for b in range(self.B):
                pos_keys = keys[b]
                # Find indices where key repeats
                _, counts = np.unique(pos_keys, return_counts=True)
                # Mark positions that appear more than once as collided
                key_to_count = {}
                for i in range(self.N):
                    key_to_count[int(pos_keys[i])] = key_to_count.get(int(pos_keys[i]), 0) + 1
                for i in range(self.N):
                    if key_to_count[int(pos_keys[i])] > 1:
                        bumped[b, i] = True

        valid_move = ~bumped & ~is_obs
        # Apply valid moves
        new_y = np.where(valid_move, py, self.agent_pos[..., 0])
        new_x = np.where(valid_move, px, self.agent_pos[..., 1])
        self.agent_pos[..., 0] = new_y
        self.agent_pos[..., 1] = new_x
        self.moved_mask = valid_move | ~bumped  # re-apply FOV for bumped agents (they didn't move but we still apply to be safe — actually for bumped we don't need to re-apply. Let's simplify: always apply FOV for all agents.)
        self.moved_mask[:] = True  # Apply FOV for everyone each step

        # ----- FOV & Coverage -----
        self.observed_now[:] = False
        self._apply_fov()

        # Coverage delta (newly visited traversable cells)
        traversable = (self.maps == 0)
        newly_covered = self.visited_global & ~prev_visited_global & traversable  # (B, G, G)
        n_new = newly_covered.sum(axis=(1, 2))                    # (B,)
        delta_cov = n_new / np.maximum(self.n_traversable, 1)      # (B,)

        # Overlap: cells in FOV this step that were already globally covered
        overlap = (self.observed_now & prev_visited_global).sum(axis=(1, 2))
        overlap_frac = overlap / np.maximum(self.n_traversable, 1)

        # Coverage & T95
        coverage = (self.visited_global & traversable).sum(axis=(1, 2)) / np.maximum(self.n_traversable, 1)
        reached_95 = (coverage >= self.cfg.success_threshold) & (self.t95 < 0)
        self.t95 = np.where(reached_95, self.step_count, self.t95)

        # ----- Communication -----
        self._update_in_range()
        # Compute bandwidth cost: cost per step = (n_send_t * d_m) / (N * d_m_max)
        # We don't know d_m here — it's a hyperparam. We approximate with 1 message = 1 cost unit.
        # Normalise by max possible sends (N per env).
        n_send = send_flags.sum(axis=1).astype(np.float32)         # (B,)
        comm_cost = n_send / self.N                                # in [0, 1]

        # ----- Reward (shared across agents in an env) -----
        r = (self.cfg.w_coverage * delta_cov
             - self.cfg.w_overlap * overlap_frac
             - self.cfg.w_time
             # Change this:
             # - self.cfg.w_comm * comm_cost
             # To this:
             - getattr(self, "comm_cost_scale", 1.0) * self.cfg.w_comm * comm_cost
             + getattr(self, "send_bonus", 0.0) * comm_cost  # Incentivize talking during bootcamp
             + self.cfg.wall_bump_penalty * bumped.sum(axis=1).astype(np.float32) / self.N)
        # Shape as (B, N) — same value for all agents in an env
        rewards = np.broadcast_to(r[:, None], (self.B, self.N)).astype(np.float32)

        # ----- Termination -----
        self.step_count += 1
        dones = (self.step_count >= self.cfg.max_steps)              # (B,)
        # Success bonus at final step
        success = reached_95 | ((self.step_count == self.cfg.max_steps) & (coverage >= self.cfg.success_threshold))
        rewards = rewards + (success[:, None] * self.cfg.success_bonus).astype(np.float32)

        # ----- Received messages -----
        # Build received message embeddings for each agent
        if message_embeddings is None:
            recv_emb = np.zeros((self.B, self.N, self.cfg.message_embed_dim), dtype=np.float32)
            recv_send = np.zeros((self.B, self.N, self.N), dtype=bool)
        else:
            recv_emb, recv_send = self._aggregate_received_messages(
                message_embeddings, send_flags
            )

        # ----- Build observation for NEXT step -----
        # NOTE: after taking the step, we observe the result and any messages received.
        self._current_obs = self._build_observation(recv_emb, recv_send)

        # Info dict
        infos = {
            "coverage": coverage,
            "t95": self.t95.copy(),
            "n_traversable": self.n_traversable.copy(),
            "comm_cost": comm_cost,
            "overlap": overlap_frac,
            "in_range_pairs": self.in_range.sum(axis=(1, 2)) / 2.0,  # undirected pairs
        }

        return self._current_obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply_fov(self) -> None:
        """Mark FOV cells as visited (local and global) for every agent."""
        pos_y = self.agent_pos[..., 0]   # (B, N)
        pos_x = self.agent_pos[..., 1]

        for dy, dx in self._fov_offsets:
            ty = pos_y + dy
            tx = pos_x + dx
            valid = (ty >= 0) & (ty < self.G) & (tx >= 0) & (tx < self.G)
            ty_c = np.clip(ty, 0, self.G - 1)
            tx_c = np.clip(tx, 0, self.G - 1)
            # Not an obstacle
            not_obs = (self.maps[self._b_idx, ty_c, tx_c] == 0)
            mask = valid & not_obs  # (B, N)

            # Scatter into visited_local (B, N, G, G)
            if mask.any():
                self.visited_local[self._b_idx[mask], self._n_idx[mask],
                                   ty_c[mask], tx_c[mask]] = True
                self.visited_global[self._b_idx[mask], ty_c[mask], tx_c[mask]] = True
                self.observed_now[self._b_idx[mask], ty_c[mask], tx_c[mask]] = True

    def _update_in_range(self) -> None:
        """Update Manhattan-distance-based comm range mask (B, N, N)."""
        # Pairwise Manhattan distance
        # agent_pos: (B, N, 2)
        pi = self.agent_pos[:, :, None, :]   # (B, N, 1, 2)
        pj = self.agent_pos[:, None, :, :]   # (B, 1, N, 2)
        dist = np.abs(pi - pj).sum(axis=-1)  # (B, N, N)
        self.in_range = (dist <= self.cfg.comm_range)
        # Don't allow self-communication
        eye = np.eye(self.N, dtype=bool)
        self.in_range[:, eye] = False

    def _aggregate_received_messages(
        self,
        message_embeddings: np.ndarray,   # (B, N, d_code)
        send_flags: np.ndarray,           # (B, N)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean-pooled received message embedding per agent.

        Returns:
            recv_emb: (B, N, d_code) float
            recv_send: (B, N, N) bool — True if agent j transmitted and i can hear j
        """
        d_code = message_embeddings.shape[-1]
        # Mask out messages from agents that didn't send
        send_mask = send_flags[:, None, :]                         # (B, 1, N)
        # in_range: (B, N, N) — True if j in range of i
        hear = self.in_range & send_mask                           # (B, N, N)
        # Aggregate: mean of sender embeddings, zero if no senders
        # message_embeddings: (B, N, d_code) — per sender
        # Broadcast to (B, recv, send, d_code) and mask
        msg = message_embeddings[:, None, :, :].repeat(self.N, axis=1)  # (B, N, N, d_code)
        masked = msg * hear[..., None].astype(np.float32)
        n_heard = hear.sum(axis=2, keepdims=True).clip(min=1)      # (B, N, 1)
        recv_emb = masked.sum(axis=2) / n_heard                    # (B, N, d_code)
        recv_emb = recv_emb.astype(np.float32)
        recv_send = hear
        return recv_emb, recv_send

    def _build_observation(
        self,
        received_embeddings: np.ndarray,   # (B, N, d_code)
        received_send_flags: np.ndarray,   # (B, N, N)
    ) -> dict[str, np.ndarray]:
        """Construct observation dict from current state."""
        B, N, G = self.B, self.N, self.G
        k = self.k
        r = self.r

        # Local maps & visited_self — extract k×k windows
        pos_y = self.agent_pos[..., 0]    # (B, N)
        pos_x = self.agent_pos[..., 1]

        # Build local maps by iterating over FOV offsets (same as _apply_fov logic, but read)
        local_map = np.full((B, N, k, k), -1.0, dtype=np.float32)    # -1 = unseen/out of bounds
        visited_self = np.zeros((B, N, k, k), dtype=np.float32)

        for i, (dy, dx) in enumerate(self._fov_offsets):
            ky = i // k
            kx = i % k
            ty = pos_y + dy
            tx = pos_x + dx
            valid = (ty >= 0) & (ty < G) & (tx >= 0) & (tx < G)
            ty_c = np.clip(ty, 0, G - 1)
            tx_c = np.clip(tx, 0, G - 1)
            is_obs = (self.maps[self._b_idx, ty_c, tx_c] == 1)
            # -1 if invalid, 1 if obstacle, 0 if free
            local_map[:, :, ky, kx] = np.where(valid,
                                               np.where(is_obs, 1.0, 0.0),
                                               -1.0)
            visited_self[:, :, ky, kx] = (
                valid & self.visited_local[self._b_idx, self._n_idx, ty_c, tx_c]
            ).astype(np.float32)

        obs = {"local_map": local_map, "visited_self": visited_self}

        if self.cfg.obs_self_pos:
            obs["self_pos"] = self.agent_pos.astype(np.float32) / self.G  # (B, N, 2)

        if self.cfg.obs_neighbours:
            # For each agent i, features for each other agent j ≠ i
            # Features: [in_range, rel_dy/G, rel_dx/G, sent_this_step]
            feats = np.zeros((B, N, N - 1, 4), dtype=np.float32)
            for i in range(N):
                idx = 0
                for j in range(N):
                    if j == i:
                        continue
                    rel_dy = (self.agent_pos[:, j, 0] - self.agent_pos[:, i, 0]).astype(np.float32) / self.G  # (B,)
                    rel_dx = (self.agent_pos[:, j, 1] - self.agent_pos[:, i, 1]).astype(np.float32) / self.G
                    in_range_ij = self.in_range[:, i, j]                    # (B,)
                    # Did j send and can i hear? (use latest send flags from last step)
                    # We don't have send_flags here — fill 0 (caller can override)
                    sent_ij = np.zeros((B,), dtype=np.float32)
                    feats[:, i, idx, 0] = in_range_ij
                    feats[:, i, idx, 1] = rel_dy
                    feats[:, i, idx, 2] = rel_dx
                    feats[:, i, idx, 3] = sent_ij
                    idx += 1
            obs["neighbour_feats"] = feats

        if self.cfg.obs_message_embed:
            # Concatenate mean-pooled embedding with sender count
            n_heard = received_send_flags.sum(axis=2, keepdims=True).astype(np.float32) / N  # (B, N, 1)
            msg_obs = np.concatenate([received_embeddings, n_heard], axis=2)  # (B, N, d_code+1)
            obs["message_embed"] = msg_obs.astype(np.float32)

        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_global_state(self) -> dict[str, np.ndarray]:
        """Return the centralised critic state."""
        traversable = (self.maps == 0).astype(np.float32)      # (B, G, G)
        visited = self.visited_global.astype(np.float32)       # (B, G, G)
        occupancy = np.zeros((self.B, self.G, self.G), dtype=np.float32)
        for n in range(self.N):
            occupancy[self._b_idx[:, n], self.agent_pos[:, n, 0], self.agent_pos[:, n, 1]] = 1.0
        state_map = np.stack([traversable, visited, occupancy], axis=1)  # (B, 3, G, G)
        positions = self.agent_pos.astype(np.float32) / self.G            # (B, N, 2)
        return {"state_map": state_map, "agent_positions": positions}

    def get_comm_mask(self) -> np.ndarray:
        """Return current in-range matrix (B, N, N) bool."""
        return self.in_range.copy()

    def close(self) -> None:
        pass
