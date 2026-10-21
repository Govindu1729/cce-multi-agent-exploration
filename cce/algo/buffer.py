"""Rollout buffer for MAPPO with GAE computation."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-length rollout buffer for multi-agent PPO.

    Stores B parallel envs × N agents × T timesteps of data.

    Supports both recurrent (GRU) and non-recurrent (MLP) modes.
    Also stores global state for centralised critic evaluation during update.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        rollout_length: int,
        obs_spec: dict[str, tuple[int, ...]],
        state_spec: dict[str, tuple[int, ...]] | None = None,
        gru_hidden_dim: int = 0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.B = num_envs
        self.N = num_agents
        self.T = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.gru_hidden_dim = gru_hidden_dim

        # Allocate observation storage
        self.obs: dict[str, np.ndarray] = {}
        for k, shape in obs_spec.items():
            self.obs[k] = np.zeros((self.T, self.B, self.N, *shape), dtype=np.float32)

        # Allocate state storage (for centralised critic)
        self.state: dict[str, np.ndarray] = {}
        if state_spec is not None:
            for k, shape in state_spec.items():
                # State is per-env (not per-agent), so shape is (T, B, *shape)
                self.state[k] = np.zeros((self.T, self.B, *shape), dtype=np.float32)

        self.actions_move = np.zeros((self.T, self.B, self.N), dtype=np.int64)
        self.actions_send = np.zeros((self.T, self.B, self.N), dtype=np.int64)
        self.log_probs = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.rewards = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.values = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.returns = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.advantages = np.zeros((self.T, self.B, self.N), dtype=np.float32)
        self.dones = np.zeros((self.T, self.B), dtype=np.float32)
        self.masks = np.ones((self.T, self.B), dtype=np.float32)  # 1 - dones

        # Comm stats for monitoring
        self.comm_costs = np.zeros((self.T, self.B), dtype=np.float32)
        self.send_counts = np.zeros((self.T, self.B), dtype=np.int64)

        # For recurrent mode
        if gru_hidden_dim > 0:
            self.rnn_hiddens = np.zeros((self.T, self.B, self.N, gru_hidden_dim), dtype=np.float32)
        else:
            self.rnn_hiddens = None

        self.step = 0

    def insert(
        self,
        obs: dict[str, np.ndarray],
        actions_move: np.ndarray,
        actions_send: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        comm_cost: np.ndarray,
        state: dict[str, np.ndarray] | None = None,
        rnn_hidden: np.ndarray | None = None,
    ) -> None:
        """Insert one timestep of data."""
        t = self.step
        for k, v in obs.items():
            self.obs[k][t] = v
        if state is not None and self.state:
            for k, v in state.items():
                if k in self.state:
                    self.state[k][t] = v
        self.actions_move[t] = actions_move
        self.actions_send[t] = actions_send
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones.astype(np.float32)
        self.masks[t] = (1.0 - dones).astype(np.float32)
        self.comm_costs[t] = comm_cost
        self.send_counts[t] = actions_send.sum(axis=-1)

        if self.rnn_hiddens is not None and rnn_hidden is not None:
            self.rnn_hiddens[t] = rnn_hidden

        self.step = (self.step + 1) % self.T

    def compute_returns(self, next_value: np.ndarray, next_done: np.ndarray) -> None:
        """Compute GAE advantages and returns.

        Args:
            next_value: (B, N) — value at T (for bootstrapping)
            next_done:  (B,) bool — whether next state is terminal
        """
        lastgael = np.zeros((self.B, self.N), dtype=np.float32)
        next_mask = (1.0 - next_done).astype(np.float32)[:, None]  # (B, 1) broadcast to (B, N)

        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_val = next_value
                next_non_terminal = next_mask
            else:
                next_val = self.values[t + 1]
                next_non_terminal = self.masks[t + 1][:, None]

            delta = (
                self.rewards[t]
                + self.gamma * next_val * next_non_terminal
                - self.values[t]
            )
            lastgael = delta + self.gamma * self.gae_lambda * next_non_terminal * lastgael
            self.advantages[t] = lastgael

        self.returns = self.advantages + self.values

    def get_batches(
        self,
        batch_size: int,
        num_epochs: int,
        shuffle: bool = True,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield mini-batches for PPO update."""
        total_size = self.T * self.B * self.N

        for _ in range(num_epochs):
            if shuffle:
                indices = np.random.permutation(total_size)
            else:
                indices = np.arange(total_size)

            for start in range(0, total_size, batch_size):
                end = min(start + batch_size, total_size)
                batch_idx = indices[start:end]

                # Convert flat index to (t, b, n)
                t_idx = batch_idx // (self.B * self.N)
                b_idx = (batch_idx // self.N) % self.B
                n_idx = batch_idx % self.N

                batch = {}
                # Observations (per-agent)
                for k, v in self.obs.items():
                    batch[f"obs_{k}"] = torch.tensor(v[t_idx, b_idx, n_idx], device=self.device)
                # State (per-env, broadcast to per-agent via indexing)
                if self.state:
                    state_batch = {}
                    for k, v in self.state.items():
                        # v has shape (T, B, *shape), index with (t_idx, b_idx)
                        state_batch[k] = torch.tensor(v[t_idx, b_idx], device=self.device)
                    batch["state"] = state_batch
                batch["actions_move"] = torch.tensor(self.actions_move[t_idx, b_idx, n_idx], device=self.device)
                batch["actions_send"] = torch.tensor(self.actions_send[t_idx, b_idx, n_idx], device=self.device)
                batch["log_probs_old"] = torch.tensor(self.log_probs[t_idx, b_idx, n_idx], device=self.device)
                batch["returns"] = torch.tensor(self.returns[t_idx, b_idx, n_idx], device=self.device)
                batch["advantages"] = torch.tensor(self.advantages[t_idx, b_idx, n_idx], device=self.device)
                batch["values_old"] = torch.tensor(self.values[t_idx, b_idx, n_idx], device=self.device)

                if self.rnn_hiddens is not None:
                    batch["rnn_hidden"] = torch.tensor(
                        self.rnn_hiddens[t_idx, b_idx, n_idx], device=self.device
                    )

                yield batch

        self.step = 0  # Reset for next rollout
