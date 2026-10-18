"""
Strict Token Budget Enforcement for Multi-Agent Communication.

This module implements a hard budget constraint on communication tokens.
When tokens are exhausted, the send action is physically masked to -inf
so the agent cannot choose to communicate.

No soft penalties are used - the budget is strictly enforced at the
action selection level.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TokenBucket:
    """
    Strict token bucket for communication budget enforcement.
    
    Each episode starts with a fixed budget of tokens. Each send action
    consumes tokens. When budget reaches zero, the send action is masked
    to -inf in the actor's logits, making it impossible to select.
    
    This is a HARD constraint - not a soft penalty in the reward.
    """
    
    def __init__(
        self,
        max_tokens: int,
        tokens_per_send: int = 1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the token bucket.
        
        Args:
            max_tokens: Maximum tokens available per episode
            tokens_per_send: Tokens consumed per send action (default: 1)
            device: Torch device for tensor operations
        """
        self.max_tokens = max_tokens
        self.tokens_per_send = tokens_per_send
        self.device = device or torch.device('cpu')
        
        # Current state
        self._remaining = max_tokens
        self._episode_step = 0
    
    def reset(self, max_tokens: Optional[int] = None):
        """Reset bucket for a new episode."""
        if max_tokens is not None:
            self.max_tokens = max_tokens
        self._remaining = self.max_tokens
        self._episode_step = 0
    
    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return self._remaining
    
    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self._remaining <= 0
    
    @property
    def fraction_remaining(self) -> float:
        """Get fraction of budget remaining (0.0 to 1.0)."""
        return max(0.0, self._remaining / self.max_tokens)
    
    def can_send(self) -> bool:
        """Check if a send action is possible with current budget."""
        return self._remaining >= self.tokens_per_send
    
    def consume(self, num_sends: int = 1) -> bool:
        """
        Consume tokens for send actions.
        
        Args:
            num_sends: Number of send actions to account for
            
        Returns:
            True if consumption succeeded, False if insufficient tokens
        """
        cost = num_sends * self.tokens_per_send
        if self._remaining >= cost:
            self._remaining -= cost
            return True
        return False
    
    def step(self):
        """Advance episode step counter."""
        self._episode_step += 1
    
    def get_mask_value(self) -> float:
        """Get the mask value for exhausted budget (-inf)."""
        return float('-inf')
    
    def state_dict(self) -> dict:
        """Get serializable state."""
        return {
            'max_tokens': self.max_tokens,
            'tokens_per_send': self.tokens_per_send,
            'remaining': self._remaining,
            'episode_step': self._episode_step
        }
    
    def load_state_dict(self, state: dict):
        """Load state from dict."""
        self.max_tokens = state['max_tokens']
        self.tokens_per_send = state['tokens_per_send']
        self._remaining = state['remaining']
        self._episode_step = state['episode_step']


def apply_communication_mask(
    logits: torch.Tensor,
    bucket: TokenBucket,
    send_action_idx: int = 1
) -> torch.Tensor:
    """
    Apply hard budget mask to action logits.
    
    If the token bucket is exhausted, the send action logit is set to -inf,
    making it impossible for the agent to select the send action.
    
    Args:
        logits: Action logits tensor of shape (batch, num_actions) or (num_actions,)
        bucket: TokenBucket instance tracking communication budget
        send_action_idx: Index of the send action in the action space
        
    Returns:
        Masked logits tensor (same shape as input)
    """
    masked_logits = logits.clone()
    
    if bucket.is_exhausted:
        # Mask send action to -inf when budget is exhausted
        if masked_logits.dim() == 1:
            masked_logits[send_action_idx] = bucket.get_mask_value()
        else:
            masked_logits[:, send_action_idx] = bucket.get_mask_value()
    
    return masked_logits


class BudgetAwareActor(nn.Module):
    """
    Actor wrapper that enforces hard communication budget.
    
    This wraps any actor network and automatically masks the send
    action when the token budget is exhausted.
    """
    
    def __init__(
        self,
        base_actor: nn.Module,
        token_bucket: TokenBucket,
        send_action_idx: int = 1
    ):
        """
        Initialize budget-aware actor.
        
        Args:
            base_actor: The underlying actor network
            token_bucket: TokenBucket for budget tracking
            send_action_idx: Index of send action in action space
        """
        super().__init__()
        self.base_actor = base_actor
        self.token_bucket = token_bucket
        self.send_action_idx = send_action_idx
    
    def forward(self, obs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with budget masking applied.
        
        Args:
            obs: Observation tensor
            *args, **kwargs: Additional arguments passed to base_actor
            
        Returns:
            Action logits with send masked if budget exhausted
        """
        logits = self.base_actor(obs, *args, **kwargs)
        return apply_communication_mask(logits, self.token_bucket, self.send_action_idx)
    
    def reset_budget(self, max_tokens: Optional[int] = None):
        """Reset the token budget for a new episode."""
        self.token_bucket.reset(max_tokens)
    
    def step_budget(self):
        """Step the budget (e.g., for token replenishment schedules)."""
        self.token_bucket.step()


def create_token_bucket_from_config(config: dict, device: Optional[torch.device] = None) -> TokenBucket:
    """
    Create TokenBucket from configuration dictionary.
    
    Expected config keys:
        - max_tokens: int, maximum tokens per episode
        - tokens_per_send: int, optional, tokens per send (default: 1)
    
    Args:
        config: Configuration dictionary
        device: Torch device
        
    Returns:
        TokenBucket instance
    """
    return TokenBucket(
        max_tokens=config['max_tokens'],
        tokens_per_send=config.get('tokens_per_send', 1),
        device=device
    )


if __name__ == "__main__":
    # Simple test
    print("Testing TokenBucket...")
    
    bucket = TokenBucket(max_tokens=5, tokens_per_send=1)
    
    # Test initial state
    assert bucket.remaining == 5
    assert not bucket.is_exhausted
    assert bucket.can_send()
    
    # Consume tokens
    for i in range(5):
        assert bucket.consume(1)
        assert bucket.remaining == 5 - i - 1
    
    # Should be exhausted now
    assert bucket.is_exhausted
    assert not bucket.can_send()
    assert not bucket.consume(1)
    
    # Test masking
    logits = torch.tensor([0.0, 1.0, 2.0])  # [no_send, send, other]
    masked = apply_communication_mask(logits, bucket, send_action_idx=1)
    assert masked[1] == float('-inf')
    assert masked[0] == 0.0
    assert masked[2] == 2.0
    
    print("All TokenBucket tests passed!")