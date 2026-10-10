"""Observation encoder producing a latent representation h_i^t."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """Encode a dict observation into a single latent vector h_i^t.

    Supports both MLP and GRU backbones. The GRU variant carries a hidden
    state across timesteps for partial observability.

    Output is a single hidden vector of dim `hidden_dim`, which is fed to:
      - the action heads (move + send)
      - the message encoder (which produces the VQ input)
    """

    def __init__(
        self,
        obs_spec: dict[str, tuple[int, ...]],
        hidden_dim: int = 64,
        backbone: str = "mlp",
        gru_hidden: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.gru_hidden = gru_hidden
        self.obs_spec = obs_spec

        # Flatten spec into total input dim
        flat_dim = sum(int(torch.tensor(shape).prod().item()) for shape in obs_spec.values())
        self.flat_dim = flat_dim

        self.input_proj = nn.Linear(flat_dim, hidden_dim)

        if backbone == "gru":
            self.gru = nn.GRUCell(hidden_dim, gru_hidden)
            self.output_proj = nn.Linear(gru_hidden, hidden_dim)
        else:
            # Two-layer MLP
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0)
        # Smaller init for output projections
        if self.backbone == "gru":
            nn.init.orthogonal_(self.output_proj.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.mlp[-2].weight, gain=0.01)

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Initial hidden state for GRU."""
        if self.backbone != "gru":
            return torch.zeros(batch_size, 0)
        if device is None:
            device = self.input_proj.weight.device
        return torch.zeros(batch_size, self.gru_hidden, device=device)

    def flatten_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten observation dict into (..., flat_dim) tensor.

        Assumes obs tensors have shape (..., *obs_spec[key]) where the leading
        dimensions are batch/agent and the trailing dimensions are features.
        Only the trailing (feature) dimensions are flattened.
        """
        parts = []
        # Maintain a fixed order (same as obs_spec)
        for key in sorted(obs.keys()):
            v = obs[key]
            # Flatten the last len(obs_spec[key]) dimensions (feature dims only)
            n_feat_dims = len(self.obs_spec[key])
            if n_feat_dims > 1:
                parts.append(v.flatten(start_dim=-n_feat_dims))
            elif n_feat_dims == 1:
                parts.append(v)
            else:
                parts.append(v.unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: dict of tensors with shape (..., *obs_spec[k])
            hidden: GRU hidden state (optional, only used for GRU backbone)
                    Shape: (..., gru_hidden) where ... matches obs leading dims.

        Returns:
            h: (..., hidden_dim)
            hidden_new: updated hidden state (zeros for MLP)
        """
        x = self.flatten_obs(obs)
        x = torch.relu(self.input_proj(x))

        if self.backbone == "gru":
            assert hidden is not None
            # GRUCell expects 2D input: (batch, hidden_dim)
            # x has shape (..., hidden_dim), hidden has shape (..., gru_hidden)
            # Flatten leading dimensions, apply GRU, reshape back
            original_shape = x.shape[:-1]  # e.g., (B, N)
            x_flat = x.reshape(-1, x.shape[-1])  # (B*N, hidden_dim)
            h_flat = hidden.reshape(-1, hidden.shape[-1])  # (B*N, gru_hidden)
            h_flat = self.gru(x_flat, h_flat)
            h = h_flat.reshape(*original_shape, self.gru_hidden)
            out = torch.relu(self.output_proj(h))
            return out, h
        else:
            out = self.mlp(x)
            dummy = torch.zeros(x.shape[:-1] + (0,), device=x.device)
            return out, dummy
