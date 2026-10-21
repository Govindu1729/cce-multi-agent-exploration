"""Communication channel: message production and aggregation with optional noise."""

from __future__ import annotations

import torch
import torch.nn as nn


class MessageEncoder(nn.Module):
    """Produce the continuous latent z that gets quantised by VQ.

    Takes the observation encoder's hidden state h_i^t and projects it
    to (n_slots, code_dim) — one slot per message position.
    """

    def __init__(self, input_dim: int, n_slots: int, code_dim: int, hidden: int = 64):
        super().__init__()
        self.n_slots = n_slots
        self.code_dim = code_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_slots * code_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (..., input_dim) — encoder hidden state.

        Returns:
            z: (..., n_slots, code_dim) — pre-quantisation latent.
        """
        z = self.net(h)
        return z.reshape(*z.shape[:-1], self.n_slots, self.code_dim)


class MessageDecoder(nn.Module):
    """Aggregate received message embeddings into a single vector for the actor.

    Mean-pools received embeddings and concatenates the sender count.
    """

    def __init__(self, code_dim: int, output_dim: int):
        super().__init__()
        # Input: code_dim (mean-pooled embedding) + 1 (sender count)
        self.proj = nn.Linear(code_dim + 1, output_dim)
        nn.init.orthogonal_(self.proj.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, recv_embeddings: torch.Tensor, sender_count: torch.Tensor) -> torch.Tensor:
        """
        Args:
            recv_embeddings: (..., code_dim) — mean-pooled received message embedding.
            sender_count: (..., 1) — number of senders heard (normalised).

        Returns:
            (..., output_dim) — aggregated message feature.
        """
        x = torch.cat([recv_embeddings, sender_count], dim=-1)
        return torch.relu(self.proj(x))
