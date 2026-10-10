"""Budget controller: Lagrangian dual mechanism for enforcing normalised β.

Maintains a running estimate of the send-threshold τ (in policy mode) or the
Lagrangian multiplier λ (in utility mode). The goal is to ensure that the
per-episode bandwidth cost equals the target β.

Two modes:
  - 'policy': agents sample a send decision from the policy. The budget
              controller modulates the send logit with -λ, where λ is
              updated by a PID-like controller based on the bandwidth error.
  - 'utility': agents output a utility u ∈ [0,1]. A message is sent iff
               u > τ, where τ is set by the controller.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class BudgetController(nn.Module):
    def __init__(
        self,
        mode: Literal["policy", "utility"] = "policy",
        target_beta: float = 0.25,
        pid_kp: float = 0.5,
        pid_ki: float = 0.05,
        pid_kd: float = 0.1,
        max_lambda: float = 10.0,
    ):
        super().__init__()
        self.mode = mode
        self.target_beta = target_beta
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.max_lambda = max_lambda

        # Non-learnable state (buffers)
        self.register_buffer("lam", torch.tensor(0.0))
        self.register_buffer("integral", torch.tensor(0.0))
        self.register_buffer("prev_error", torch.tensor(0.0))
        self.register_buffer("episode_cost", torch.tensor(0.0))
        self.register_buffer("episode_steps", torch.tensor(0, dtype=torch.long))

    def reset_episode(self) -> None:
        """Reset per-episode accumulators. Call at episode start."""
        self.episode_cost.zero_()
        self.episode_steps.zero_()
        self.integral.zero_()
        self.prev_error.zero_()

    def on_step(self, comm_cost: float | torch.Tensor) -> None:
        """Record per-step cost for the running episode average."""
        with torch.no_grad():
            self.episode_cost.add_(comm_cost)
            self.episode_steps.add_(1)

    def on_episode_end(self) -> dict[str, float]:
        """Update λ (or τ) at the end of an episode based on average cost."""
        with torch.no_grad():
            steps = self.episode_steps.item()
            if steps < 1:
                avg_cost = 0.0
            else:
                avg_cost = self.episode_cost.item() / steps
            error = avg_cost - self.target_beta

            # PID update
            P = self.pid_kp * error
            self.integral.add_(error)
            I = self.pid_ki * self.integral.item()
            D = self.pid_kd * (error - self.prev_error.item())
            delta = P + I + D

            if self.mode == "policy":
                self.lam.add_(delta)
                # Change this:
                # self.lam.clamp_(0.0, self.max_lambda)
                # To this:
                self.lam.clamp_(min=-1.0, max=self.max_lambda)  # ALLOW SUBSIDY
            else:
                # Utility mode: map lambda to a threshold in [0, 1]
                self.lam.add_(delta)
                self.lam.clamp_(0.0, 1.0)

            self.prev_error.copy_(torch.tensor(error))

            return {
                "lambda": self.lam.item(),
                "actual_beta": avg_cost,
                "target_beta": self.target_beta,
                "error": error,
            }

    def get_send_logit_bias(self) -> torch.Tensor:
        """Return the logit bias to subtract from the send logit (policy mode)."""
        return self.lam

    def get_threshold(self) -> torch.Tensor:
        """Return the threshold τ (utility mode)."""
        return self.lam

    def decide_send_utility(
        self,
        utility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Utility-mode: send iff u > τ, with straight-through gradient.

        Args:
            utility: (...,) in [0, 1].

        Returns:
            hard_gate: (...,) bool — final send decision.
            soft_gate: (...,) float — differentiable soft gate for gradient flow.
        """
        tau = self.get_threshold()
        soft = torch.sigmoid((utility - tau) / 0.1)
        hard = (utility > tau).float()
        # Straight-through
        gate = hard + (soft - soft.detach())
        return hard.bool(), gate
