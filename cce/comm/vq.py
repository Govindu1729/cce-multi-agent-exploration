"""Vector Quantizer with EMA codebook updates, commitment loss, and dead-code revival.

Based on VQ-VAE (van den Oord et al., 2017) with EMA updates from DeepMind's
implementation and dead-code revival inspired by Liu et al., 2022 (ADCB).

Output: a sequence of `n_slots` discrete codebook indices. Each code is an
embedding of dim `code_dim`. The total message is thus n_slots codes × log2(K) bits.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_codes: int = 16,
        code_dim: int = 8,
        n_slots: int = 4,
        decay: float = 0.99,
        commitment: float = 0.25,
        eps: float = 1e-5,
        dead_reset_steps: int = 1000,
        dead_threshold: float = 0.01,
    ):
        """
        Args:
            n_codes: K, number of codes in the codebook.
            code_dim: dimensionality of each code embedding.
            n_slots: number of code slots per message (d_m in the paper).
            decay: EMA decay for codebook updates.
            commitment: weight on the commitment loss.
            eps: Laplace smoothing constant.
            dead_reset_steps: check for dead codes every N calls to `forward`.
            dead_threshold: a code is "dead" if used < dead_threshold fraction of the time.
        """
        super().__init__()
        self.K = n_codes
        self.D = code_dim
        self.S = n_slots
        self.decay = decay
        self.commitment = commitment
        self.eps = eps
        self.dead_reset_steps = dead_reset_steps
        self.dead_threshold = dead_threshold

        # Codebook: (K, D)
        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

        # EMA state (not parameters, just buffers)
        self.register_buffer("ema_cluster_size", torch.zeros(n_codes, dtype=torch.float32))
        self.register_buffer("ema_embedding_sum", self.codebook.weight.clone())
        self.register_buffer("usage_counter", torch.zeros(n_codes, dtype=torch.float32))
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.int64))

    @property
    def bits_per_message(self) -> int:
        import math
        return self.S * int(math.ceil(math.log2(max(self.K, 2))))

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantise continuous latent z to discrete codes.

        Args:
            z: (..., n_slots, code_dim) continuous latent.

        Returns:
            dict with:
              - 'quantised': (..., n_slots, code_dim) — quantised embeddings (detached in forward)
              - 'indices': (..., n_slots) int64 — codebook indices
              - 'loss': scalar — VQ loss (commitment + codebook EMA loss)
              - 'perplexity': scalar — codebook usage perplexity
        """
        assert z.shape[-2:] == (self.S, self.D), f"Expected shape (..., {self.S}, {self.D}), got {z.shape}"

        flat_z = z.reshape(-1, self.D)  # (M, D) where M = prod(batch) * S
        M = flat_z.shape[0]

        # Distances to codebook
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        z2 = (flat_z ** 2).sum(dim=1, keepdim=True)                    # (M, 1)
        e2 = (self.codebook.weight ** 2).sum(dim=1, keepdim=True).T    # (1, K)
        z_e = flat_z @ self.codebook.weight.T                           # (M, K)
        dist = z2 + e2 - 2 * z_e                                        # (M, K)

        indices = dist.argmin(dim=1)                                     # (M,)

        # Straight-through estimator
        quantised_flat = self.codebook(indices)                          # (M, D)
        # ST: forward = quantised, backward = identity through z
        quantised_flat_st = flat_z + (quantised_flat - flat_z).detach()

        # --- EMA codebook update (only during training) ---
        if self.training:
            # Update usage counter
            self.usage_counter.scatter_add_(
                0, indices, torch.ones_like(indices, dtype=torch.float32)
            )

            # EMA cluster size
            one_hot = F.one_hot(indices, self.K).float()                 # (M, K)
            updated_cluster_size = one_hot.sum(dim=0)                    # (K,)
            self.ema_cluster_size.mul_(self.decay).add_(
                updated_cluster_size * (1 - self.decay)
            )

            # EMA embedding sum
            updated_embedding_sum = one_hot.T @ flat_z                   # (K, D)
            self.ema_embedding_sum.mul_(self.decay).add_(
                updated_embedding_sum * (1 - self.decay)
            )

            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            smoothed_cluster = (self.ema_cluster_size + self.eps) / (n + self.K * self.eps) * n
            with torch.no_grad():
                self.codebook.weight.copy_(self.ema_embedding_sum / smoothed_cluster.unsqueeze(1))

            # Dead-code check
            self.step_counter += 1
            if self.step_counter.item() % self.dead_reset_steps == 0:
                self._maybe_reset_dead_codes(flat_z)

        # Commitment loss (encoder → codebook): ||z - sg(e)||^2
        commitment_loss = F.mse_loss(flat_z, quantised_flat.detach())

        # Codebook EMA loss (already in-place updated, but track for logging)
        # We don't need a gradient-based codebook loss since EMA is non-differentiable.

        # Reshape
        quantised = quantised_flat_st.reshape(z.shape)
        indices = indices.reshape(z.shape[:-1])  # (..., n_slots)

        # Perplexity (codebook utilisation)
        with torch.no_grad():
            # Compute usage probabilities from indices (works in both train and eval)
            one_hot_eval = F.one_hot(indices, self.K).float()
            avg_probs = one_hot_eval.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "quantised": quantised,
            "indices": indices,
            "loss": self.commitment * commitment_loss,
            "perplexity": perplexity.detach(),
        }

    def _maybe_reset_dead_codes(self, z_sample: torch.Tensor) -> None:
        """Reset codes that have been used < dead_threshold fraction of the time."""
        total = self.usage_counter.sum().item()
        if total < 1:
            return
        usage_frac = self.usage_counter / total
        dead_mask = usage_frac < self.dead_threshold
        if not dead_mask.any():
            return

        # Reset to random samples from the current z distribution
        n_dead = int(dead_mask.sum().item())
        sample_idx = torch.randint(0, z_sample.shape[0], (n_dead,), device=z_sample.device)
        new_codes = z_sample[sample_idx]
        # Add small noise for diversity
        new_codes = new_codes + 0.02 * torch.randn_like(new_codes)
        with torch.no_grad():
            self.codebook.weight[dead_mask] = new_codes
            # Reset EMA state for dead codes
            self.ema_cluster_size[dead_mask] = 0
            self.ema_embedding_sum[dead_mask] = new_codes
            self.usage_counter[dead_mask] = 0

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook embeddings from indices.

        Args:
            indices: (..., n_slots) int64 in [0, K)

        Returns:
            (..., n_slots, code_dim) float
        """
        return self.codebook(indices)

    def get_usage_stats(self) -> dict[str, float]:
        """Return current codebook usage statistics."""
        total = self.usage_counter.sum().item()
        if total < 1:
            return {"active_codes": 0, "perplexity": 0.0}
        usage_frac = self.usage_counter / total
        active = (usage_frac >= self.dead_threshold).sum().item()
        return {
            "active_codes": int(active),
            "total_codes": self.K,
            "usage_entropy": float(-(usage_frac * torch.log(usage_frac + 1e-10)).sum().item()),
        }
