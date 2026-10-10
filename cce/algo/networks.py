"""Actor and critic networks for MAPPO with communication."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cce.comm.encoder import ObsEncoder
from cce.comm.channel import MessageEncoder, MessageDecoder
from cce.comm.vq import VectorQuantizer
from cce.comm.budget import BudgetController
from cce.algo.distributions import MultiActionDist


class Actor(nn.Module):
    """Shared actor for homogeneous agents.

    Architecture:
        obs  →  ObsEncoder  →  h  →  move head (4-way categorical)
                              h  →  send head (Bernoulli)
                              h  →  MessageEncoder  →  z  →  VQ  →  message

    Returns:
        dist: MultiActionDist over (move, send)
        message_indices: (B, N, n_slots) int — VQ codebook indices
        vq_loss: scalar — VQ commitment loss
        perplexity: scalar — codebook utilisation
    """

    def __init__(
        self,
        obs_spec: dict[str, tuple[int, ...]],
        n_actions: int = 4,
        hidden_dim: int = 64,
        backbone: str = "mlp",
        gru_hidden: int = 128,
        n_codes: int = 16,
        code_dim: int = 8,
        n_slots: int = 4,
        budget_mode: str = "policy",
        target_beta: float = 0.25,
    ):
        super().__init__()
        self.obs_spec = obs_spec
        self.n_actions = n_actions
        self.n_slots = n_slots
        self.code_dim = code_dim

        self.encoder = ObsEncoder(
            obs_spec=obs_spec,
            hidden_dim=hidden_dim,
            backbone=backbone,
            gru_hidden=gru_hidden,
        )
        self.message_encoder = MessageEncoder(hidden_dim, n_slots, code_dim)
        self.vq = VectorQuantizer(n_codes=n_codes, code_dim=code_dim, n_slots=n_slots)

        # Action heads
        self.move_head = nn.Linear(hidden_dim, n_actions)
        self.send_head = nn.Linear(hidden_dim, 1)

        # Utility head (for utility budget mode)
        if budget_mode == "utility":
            self.utility_head = nn.Linear(hidden_dim, 1)
        else:
            self.utility_head = None

        self.budget_controller = BudgetController(
            mode=budget_mode, target_beta=target_beta
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.encoder.init_hidden(batch_size, device)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
    ) -> tuple[MultiActionDist, dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            obs: dict of tensors (..., *obs_spec[k])
            hidden: GRU hidden state (..., gru_hidden) or None

        Returns:
            dist: MultiActionDist over (move, send)
            info: dict with 'message_indices', 'vq_loss', 'perplexity',
                  'message_embeddings', 'message_quantised'
            hidden_new: updated GRU hidden (same shape as input)
        """
        h, hidden_new = self.encoder(obs, hidden)

        # Move head
        move_logits = self.move_head(h)  # (..., 4)
        # Send head with budget bias
        send_logit = self.send_head(h).squeeze(-1)  # (...)
        if self.budget_controller.mode == "policy":
            send_logit = send_logit - self.budget_controller.get_send_logit_bias()

        dist = MultiActionDist(move_logits, send_logit)

        # Message: quantise h → z → VQ
        z = self.message_encoder(h)  # (..., n_slots, code_dim)
        vq_out = self.vq(z)
        info = {
            "message_indices": vq_out["indices"],
            "vq_loss": vq_out["loss"],
            "perplexity": vq_out["perplexity"],
            "message_quantised": vq_out["quantised"],
        }

        return dist, info, hidden_new


class Critic(nn.Module):
    """Centralised critic operating on the global state.

    Supports MLP (flattened) or CNN (spatial) encoding of the state map.
    """

    def __init__(
        self,
        grid_size: int,
        num_agents: int,
        state_channels: int = 3,
        hidden_dim: int = 256,
        critic_type: str = "cnn",
    ):
        super().__init__()
        self.critic_type = critic_type
        self.grid_size = grid_size
        self.num_agents = num_agents

        if critic_type == "cnn":
            # Adaptive pooling to (1, 1) ensures compatibility with any grid size
            self.state_encoder = nn.Sequential(
                nn.Conv2d(state_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            # 64 * 1 * 1 = 64
            mlp_in = 64 + num_agents * 2
        else:
            # MLP: flatten the whole state
            mlp_in = state_channels * grid_size * grid_size + num_agents * 2

        self.value_head = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
        # Final layer smaller init
        last_linear = [m for m in self.value_head.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            state: dict with 'state_map' (B, C, G, G) and 'agent_positions' (B, N, 2).

        Returns:
            values: (B, 1)
        """
        state_map = state["state_map"]
        positions = state["agent_positions"]
        flat_pos = positions.reshape(positions.shape[0], -1)

        if self.critic_type == "cnn":
            x = self.state_encoder(state_map)
        else:
            x = state_map.reshape(state_map.shape[0], -1)

        x = torch.cat([x, flat_pos], dim=-1)
        return self.value_head(x)
