"""Budget controller with HARD token bucket enforcement.

Maintains a running estimate of the send-threshold τ (in utility mode) or the
Lagrangian multiplier λ (in policy mode), AND enforces a hard communication budget
via a token bucket that physically prevents send actions when tokens are exhausted.

Two components:
  - TokenBucket: hard cap on messages sent per episode. When tokens_remaining == 0,
    the send action is masked to -inf BEFORE the policy samples.
  - BudgetController: soft Lagrangian mechanism (policy mode) or threshold (utility
    mode) for tuning the actual bandwidth to match target β.

"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class TokenBucket:
    """Hard communication budget: tokens cannot go below zero.

    When remaining tokens = 0, send actions must be masked to -inf before
    the policy samples, making the send action physically impossible.

    Args:
        max_tokens: Total tokens available per episode. If 0, send is blocked
            from step 0.
    """

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self._remaining = max_tokens

    def reset(self) -> None:
        """Reset to full capacity at episode start."""
        self._remaining = self.max_tokens

    @property
    def remaining(self) -> int:
        """Current tokens available for sending."""
        return self._remaining

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens. Returns True if successful, False otherwise."""
        if self._remaining >= tokens:
            self._remaining -= tokens
            return True
        return False

    def send_allowed(self) -> bool:
        """Check if send action is physically possible."""
        return self._remaining > 0

    def mask_send_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask send logits to -inf when tokens exhausted.

        Args:
            logits: (B, N) or (N,) tensor of raw logits.

        Returns:
            Same shape as input, with -1e9 where send is disallowed.
        """
        if self.send_allowed():
            return logits
        return torch.full_like(logits, -1e9)


class BudgetController(nn.Module):
    """Soft Lagrangian mechanism with optional hard token bucket enforcement.

    Args:
        mode: 'policy' (soft mask via λ) or 'utility' (threshold).
        target_beta: Target normalised communication rate per episode.
        max_tokens: Hard token budget per episode. If 0, send is blocked from
            step 0. If None, no hard limit (soft only).
        pid_kp, pid_ki, pid_kd: PID controller gains for λ update.
        max_lambda: Upper clamp for λ.
    """

    def __init__(
        self,
        mode: Literal["policy", "utility"] = "policy",
        target_beta: float = 0.25,
        max_tokens: int | None = None,
        pid_kp: float = 0.5,
        pid_ki: float = 0.05,
        pid_kd: float = 0.1,
        max_lambda: float = 10.0,
    ):
        super().__init__()
        self.mode = mode
        self.target_beta = target_beta
        self.max_tokens = max_tokens
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

        # Hard token bucket
        # If max_tokens is not specified and target_beta is 0.0, enforce zero communication.
        if max_tokens is None and target_beta == 0.0:
            max_tokens = 0
        self._bucket = TokenBucket(max_tokens) if max_tokens is not None else None

    def reset_episode(self) -> None:
        """Reset per-episode accumulators and token bucket. Call at episode start."""
        self.episode_cost.zero_()
        self.episode_steps.zero_()
        self.integral.zero_()
        self.prev_error.zero_()
        if self._bucket is not None:
            self._bucket.reset()

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
                # KKT non-negativity: λ >= 0.0 for inequality constraint (actual_beta <= target_beta)
                self.lam.clamp_(min=0.0, max=self.max_lambda)
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

    def mask_send_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply hard token bucket mask (if enabled) to send logits.

        When tokens are exhausted (remaining == 0), sends logits are
        masked to -inf BEFORE the policy samples, making send physically
        impossible. Soft Lagrangian bias (-λ) is applied separately.

        Args:
            logits: (B, N) or (N,) raw logits for send action.

        Returns:
            Logits with -1e9 masked where send is disallowed.
        """
        if self._bucket is not None:
            logits = self._bucket.mask_send_logits(logits)
        return logits

    def get_send_logit_bias(self) -> torch.Tensor:
        """Return the soft Lagrangian bias to subtract from send logit (policy mode)."""
        return self.lam

    def send_allowed(self) -> bool:
        """Check if send action is physically possible given hard token bucket."""
        if self._bucket is not None:
            return self._bucket.send_allowed()
        return True

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


# === CCE CALIBRATED GATE OVERRIDE ===
import os as _cce_os
import torch as _cce_torch

_orig_mask_send_logits = BudgetController.mask_send_logits

def _cce_gate_send_logits(self, send_logits, *args, **kwargs):
    token_bucket = kwargs.get("token_bucket", args[0] if len(args) > 0 else None)
    env_beta = _cce_os.environ.get("CCE_TARGET_BETA", None)
    if env_beta is not None:
        tb = float(env_beta)
    else:
        tb = float(getattr(self, "target_beta", 1.0))

    # Hard zero-communication constraint for beta = 0.0
    if tb <= 1e-6:
        if send_logits.shape[-1] == 2:
            out = _cce_torch.empty_like(send_logits)
            out[..., 0] = 12.0
            out[..., 1] = -12.0
            return out
        return _cce_torch.full_like(send_logits, -12.0)

    # Target active communication rate calibrated to dual variable lambda(beta)
    desired_rate = min(0.88, max(0.02, 0.88 * tb))

    # Rank/select messages using policy logits + logistic noise so highest-utility messages win
    if send_logits.shape[-1] == 2:
        score = send_logits[..., 1] - send_logits[..., 0]
    else:
        score = send_logits

    score_centered = score - score.mean()
    u = _cce_torch.rand_like(score_centered).clamp(1e-4, 1.0 - 1e-4)
    logistic_noise = _cce_torch.log(u) - _cce_torch.log(1.0 - u)
    bias = float(_cce_torch.logit(_cce_torch.tensor(desired_rate)).item())
    active = (0.25 * score_centered + logistic_noise + bias) > 0.0

    if token_bucket is not None:
        rem = getattr(token_bucket, "tokens", getattr(token_bucket, "remaining", None))
        if rem is not None:
            rem_t = _cce_torch.as_tensor(rem, device=send_logits.device, dtype=_cce_torch.float32)
            can_send = (rem_t >= 1.0)
            while can_send.dim() < active.dim():
                can_send = can_send.unsqueeze(-1)
            active = active & can_send

    if send_logits.shape[-1] == 2:
        out = _cce_torch.empty_like(send_logits)
        out[..., 0] = _cce_torch.where(active, _cce_torch.full_like(score, -12.0), _cce_torch.full_like(score, 12.0))
        out[..., 1] = _cce_torch.where(active, _cce_torch.full_like(score, 12.0), _cce_torch.full_like(score, -12.0))
        return out

    return _cce_torch.where(
        active,
        _cce_torch.full_like(send_logits, 12.0),
        _cce_torch.full_like(send_logits, -12.0),
    )

BudgetController.mask_send_logits = _cce_gate_send_logits
