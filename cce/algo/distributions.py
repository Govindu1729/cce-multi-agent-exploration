"""Probability distributions for action sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CategoricalDist:
    """Categorical distribution with entropy bonus and log-prob utilities."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.dist = Categorical(logits=logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def mode(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)


class BernoulliDist:
    """Bernoulli distribution for the send decision."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def sample(self) -> torch.Tensor:
        probs = torch.sigmoid(self.logits)
        return (torch.rand_like(probs) < probs).long()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Numerically stable: -log(1 + exp(-logits * (2*action - 1)))
        # But simpler: use F.binary_cross_entropy_with_logits
        return -F.binary_cross_entropy_with_logits(
            self.logits, actions.float(), reduction="none"
        )

    def entropy(self) -> torch.Tensor:
        probs = torch.sigmoid(self.logits)
        return -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))

    def mode(self) -> torch.Tensor:
        return (self.logits > 0).long()


class MultiActionDist:
    """Joint distribution over (move, send) actions."""

    def __init__(self, move_logits: torch.Tensor, send_logits: torch.Tensor):
        self.move = CategoricalDist(move_logits)
        self.send = BernoulliDist(send_logits)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.move.sample(), self.send.sample()

    def log_prob(self, move_actions: torch.Tensor, send_actions: torch.Tensor) -> torch.Tensor:
        return self.move.log_prob(move_actions) + self.send.log_prob(send_actions)

    def entropy(self) -> torch.Tensor:
        return self.move.entropy() + self.send.entropy()

    def mode(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.move.mode(), self.send.mode()
