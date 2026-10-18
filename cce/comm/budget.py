"""
Decoupled Communication Budget with Granular Constraints.

This module extends the strict token budget with additional dimensions:
- max_dm: Maximum message dimension (bits per message)
- comm_frequency: Communication frequency (send allowed every N steps)
- topology: Communication topology ('fully_connected', 'nearest_neighbor', 'ring', 'star')

These constraints can be varied independently for ablation studies.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Literal, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class TopologyType(str, Enum):
    """Communication topology types."""
    FULLY_CONNECTED = "fully_connected"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    RING = "ring"
    STAR = "star"
    RANDOM = "random"


@dataclass
class CommConstraints:
    """
    Granular communication constraints for decoupled sweeps.
    
    Each constraint can be varied independently:
    - max_dm: Maximum message dimension (bits)
    - comm_frequency: Send allowed every N steps (1.0 = every step, 0.5 = every 2 steps)
    - topology: Communication graph topology
    - max_tokens: Total token budget per episode (legacy scalar budget)
    - tokens_per_send: Tokens consumed per send
    """
    max_dm: int = 4                    # Message dimension in bits
    comm_frequency: float = 1.0        # Fraction of steps where send is allowed (0 < f <= 1)
    topology: TopologyType = TopologyType.FULLY_CONNECTED
    max_tokens: int = 10               # Total token budget
    tokens_per_send: int = 1           # Tokens per send action
    comm_range: int = 3                # For nearest_neighbor topology
    
    def __post_init__(self):
        """Validate constraints."""
        if not 0 < self.comm_frequency <= 1.0:
            raise ValueError(f"comm_frequency must be in (0, 1], got {self.comm_frequency}")
        if self.max_dm <= 0:
            raise ValueError(f"max_dm must be positive, got {self.max_dm}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be non-negative, got {self.max_tokens}")
    
    @property
    def send_every_n_steps(self) -> int:
        """Convert frequency to step interval."""
        return max(1, int(1.0 / self.comm_frequency))
    
    def is_send_allowed_at_step(self, step: int) -> bool:
        """Check if sending is allowed at given step based on frequency."""
        return step % self.send_every_n_steps == 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'max_dm': self.max_dm,
            'comm_frequency': self.comm_frequency,
            'topology': self.topology.value if isinstance(self.topology, TopologyType) else self.topology,
            'max_tokens': self.max_tokens,
            'tokens_per_send': self.tokens_per_send,
            'comm_range': self.comm_range,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'CommConstraints':
        """Create from dictionary."""
        topology = d.get('topology', TopologyType.FULLY_CONNECTED)
        if isinstance(topology, str):
            topology = TopologyType(topology)
        return cls(
            max_dm=d.get('max_dm', 4),
            comm_frequency=d.get('comm_frequency', 1.0),
            topology=topology,
            max_tokens=d.get('max_tokens', 10),
            tokens_per_send=d.get('tokens_per_send', 1),
            comm_range=d.get('comm_range', 3),
        )


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


class TopologyManager:
    """
    Manages communication topology constraints.
    
    Determines which agent pairs can communicate based on topology type.
    """
    
    def __init__(
        self,
        topology: TopologyType,
        num_agents: int,
        comm_range: int = 3,
        grid_size: int = 10,
        agent_positions: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Initialize topology manager.
        
        Args:
            topology: Type of communication topology
            num_agents: Number of agents
            comm_range: Communication range (for nearest_neighbor)
            grid_size: Grid size (for position-based topologies)
            agent_positions: Current agent positions (updated each step)
        """
        self.topology = topology
        self.num_agents = num_agents
        self.comm_range = comm_range
        self.grid_size = grid_size
        self.agent_positions = agent_positions or [(0, 0)] * num_agents
        
        # Pre-compute static topologies
        self._static_edges = self._compute_static_edges()
    
    def _compute_static_edges(self) -> Set[Tuple[int, int]]:
        """Compute static communication edges based on topology."""
        edges = set()
        
        if self.topology == TopologyType.FULLY_CONNECTED:
            # All pairs can communicate
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    edges.add((i, j))
                    edges.add((j, i))
        
        elif self.topology == TopologyType.RING:
            # Ring: each agent connects to next (and previous)
            for i in range(self.num_agents):
                next_idx = (i + 1) % self.num_agents
                edges.add((i, next_idx))
                edges.add((next_idx, i))
        
        elif self.topology == TopologyType.STAR:
            # Star: agent 0 is hub, all others connect to it
            for i in range(1, self.num_agents):
                edges.add((0, i))
                edges.add((i, 0))
        
        elif self.topology == TopologyType.RANDOM:
            # Random: will be randomized each episode
            pass
        
        # NEAREST_NEIGHBOR is dynamic, computed each step
        
        return edges
    
    def update_positions(self, positions: List[Tuple[int, int]]):
        """Update agent positions for dynamic topologies."""
        self.agent_positions = positions
    
    def get_allowed_receivers(self, sender_idx: int) -> List[int]:
        """
        Get list of agent indices that sender can communicate with.
        
        Args:
            sender_idx: Index of sending agent
            
        Returns:
            List of receiver indices
        """
        allowed = []
        
        if self.topology == TopologyType.FULLY_CONNECTED:
            allowed = [j for j in range(self.num_agents) if j != sender_idx]
        
        elif self.topology == TopologyType.NEAREST_NEIGHBOR:
            # Agents within comm_range (Chebyshev distance)
            sender_pos = self.agent_positions[sender_idx]
            for j, pos in enumerate(self.agent_positions):
                if j == sender_idx:
                    continue
                dist = max(abs(sender_pos[0] - pos[0]), abs(sender_pos[1] - pos[1]))
                if dist <= self.comm_range:
                    allowed.append(j)
        
        elif self.topology == TopologyType.RING:
            next_idx = (sender_idx + 1) % self.num_agents
            prev_idx = (sender_idx - 1) % self.num_agents
            allowed = [next_idx, prev_idx]
        
        elif self.topology == TopologyType.STAR:
            if sender_idx == 0:
                # Hub can talk to everyone
                allowed = [j for j in range(1, self.num_agents)]
            else:
                # Leaves can only talk to hub
                allowed = [0]
        
        elif self.topology == TopologyType.RANDOM:
            # Random edges - should be set externally per episode
            allowed = [j for j in range(self.num_agents) if j != sender_idx]
        
        return allowed
    
    def can_communicate(self, sender_idx: int, receiver_idx: int) -> bool:
        """Check if a specific sender-receiver pair can communicate."""
        return receiver_idx in self.get_allowed_receivers(sender_idx)
    
    def randomize_edges(self, seed: Optional[int] = None):
        """Randomize edges for RANDOM topology."""
        if self.topology != TopologyType.RANDOM:
            return
        
        rng = np.random.RandomState(seed)
        self._static_edges.clear()
        
        # Create random connected graph
        # Ensure connectivity first with a random spanning tree
        nodes = list(range(self.num_agents))
        rng.shuffle(nodes)
        
        for i in range(1, self.num_agents):
            u = nodes[i]
            v = nodes[rng.randint(0, i)]
            self._static_edges.add((u, v))
            self._static_edges.add((v, u))
        
        # Add some extra random edges
        extra_edges = rng.randint(0, self.num_agents)
        for _ in range(extra_edges):
            u = rng.randint(0, self.num_agents)
            v = rng.randint(0, self.num_agents)
            if u != v:
                self._static_edges.add((u, v))
                self._static_edges.add((v, u))
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix for current topology."""
        adj = np.zeros((self.num_agents, self.num_agents), dtype=bool)
        for i in range(self.num_agents):
            for j in self.get_allowed_receivers(i):
                adj[i, j] = True
        return adj


class DecoupledCommBudget:
    """
    Unified budget manager combining all granular constraints.
    
    Enforces:
    1. Token budget (max_tokens)
    2. Message dimension limit (max_dm)
    3. Frequency constraint (comm_frequency)
    4. Topology constraint
    """
    
    def __init__(
        self,
        constraints: CommConstraints,
        num_agents: int,
        grid_size: int = 10,
        device: Optional[torch.device] = None
    ):
        """
        Initialize decoupled communication budget.
        
        Args:
            constraints: CommConstraints object with all parameters
            num_agents: Number of agents
            grid_size: Grid size for topology
            device: Torch device
        """
        self.constraints = constraints
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.device = device or torch.device('cpu')
        
        # Token bucket for budget tracking
        self.token_bucket = TokenBucket(
            max_tokens=constraints.max_tokens,
            tokens_per_send=constraints.tokens_per_send,
            device=device
        )
        
        # Topology manager
        self.topology = TopologyManager(
            topology=constraints.topology,
            num_agents=num_agents,
            comm_range=constraints.comm_range,
            grid_size=grid_size
        )
        
        # Episode state
        self._episode_step = 0
        self._agent_positions = [(0, 0)] * num_agents
    
    def reset(self, agent_positions: Optional[List[Tuple[int, int]]] = None):
        """Reset for new episode."""
        self.token_bucket.reset()
        self._episode_step = 0
        
        if agent_positions is not None:
            self._agent_positions = agent_positions
            self.topology.update_positions(agent_positions)
        
        # Randomize for random topology
        if self.constraints.topology == TopologyType.RANDOM:
            self.topology.randomize_edges()
    
    def step(self, agent_positions: Optional[List[Tuple[int, int]]] = None):
        """Advance step counter and update positions."""
        self._episode_step += 1
        self.token_bucket.step()
        
        if agent_positions is not None:
            self._agent_positions = agent_positions
            self.topology.update_positions(agent_positions)
    
    @property
    def episode_step(self) -> int:
        return self._episode_step
    
    def can_send(self, sender_idx: int, receiver_idx: int) -> bool:
        """
        Check if a specific send action is allowed.
        
        Checks all constraints:
        1. Token budget not exhausted
        2. Frequency allows sending at this step
        3. Topology allows this sender-receiver pair
        
        Args:
            sender_idx: Sending agent index
            receiver_idx: Receiving agent index
            
        Returns:
            True if send is allowed
        """
        # Check token budget
        if not self.token_bucket.can_send():
            return False
        
        # Check frequency
        if not self.constraints.is_send_allowed_at_step(self._episode_step):
            return False
        
        # Check topology
        if not self.topology.can_communicate(sender_idx, receiver_idx):
            return False
        
        return True
    
    def get_allowed_receivers(self, sender_idx: int) -> List[int]:
        """Get all allowed receivers for a sender at current step."""
        if not self.token_bucket.can_send():
            return []
        
        if not self.constraints.is_send_allowed_at_step(self._episode_step):
            return []
        
        return self.topology.get_allowed_receivers(sender_idx)
    
    def consume_token(self, sender_idx: int, num_sends: int = 1) -> bool:
        """Consume tokens for successful sends."""
        return self.token_bucket.consume(num_sends)
    
    def apply_mask(
        self,
        logits: torch.Tensor,
        sender_idx: int,
        send_action_idx: int = 1
    ) -> torch.Tensor:
        """
        Apply communication mask to logits based on all constraints.
        
        Masks send action if ANY constraint prevents sending.
        
        Args:
            logits: Action logits for sender
            sender_idx: Index of sending agent
            send_action_idx: Index of send action in action space
            
        Returns:
            Masked logits
        """
        masked_logits = logits.clone()
        
        # Check if ANY constraint prevents sending
        budget_exhausted = self.token_bucket.is_exhausted
        frequency_blocked = not self.constraints.is_send_allowed_at_step(self._episode_step)
        no_receivers = len(self.get_allowed_receivers(sender_idx)) == 0
        
        if budget_exhausted or frequency_blocked or no_receivers:
            if masked_logits.dim() == 1:
                masked_logits[send_action_idx] = float('-inf')
            else:
                masked_logits[:, send_action_idx] = float('-inf')
        
        return masked_logits
    
    def get_state(self) -> dict:
        """Get current state for logging/debugging."""
        return {
            'episode_step': self._episode_step,
            'tokens_remaining': self.token_bucket.remaining,
            'budget_exhausted': self.token_bucket.is_exhausted,
            'frequency_allowed': self.constraints.is_send_allowed_at_step(self._episode_step),
            'topology': self.constraints.topology.value,
            'adjacency_matrix': self.topology.get_adjacency_matrix().tolist(),
        }


def apply_communication_mask(
    logits: torch.Tensor,
    bucket: TokenBucket,
    send_action_idx: int = 1
) -> torch.Tensor:
    """
    Apply hard budget mask to action logits (legacy function).
    
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
    Create TokenBucket from configuration dictionary (legacy).
    
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


def create_decoupled_budget_from_config(
    config: dict,
    num_agents: int,
    grid_size: int = 10,
    device: Optional[torch.device] = None
) -> DecoupledCommBudget:
    """
    Create DecoupledCommBudget from configuration dictionary.
    
    Expected config keys:
        - max_dm: int, message dimension (default: 4)
        - comm_frequency: float, send frequency (default: 1.0)
        - topology: str, topology type (default: 'fully_connected')
        - max_tokens: int, max tokens (default: 10)
        - tokens_per_send: int, tokens per send (default: 1)
        - comm_range: int, communication range (default: 3)
    
    Args:
        config: Configuration dictionary
        num_agents: Number of agents
        grid_size: Grid size
        device: Torch device
        
    Returns:
        DecoupledCommBudget instance
    """
    constraints = CommConstraints.from_dict(config)
    return DecoupledCommBudget(
        constraints=constraints,
        num_agents=num_agents,
        grid_size=grid_size,
        device=device
    )


if __name__ == "__main__":
    # Test the new functionality
    print("Testing DecoupledCommBudget...")
    
    # Test 1: Basic constraints
    constraints = CommConstraints(
        max_dm=4,
        comm_frequency=1.0,
        topology=TopologyType.FULLY_CONNECTED,
        max_tokens=10
    )
    print(f"Constraints: {constraints.to_dict()}")
    print(f"Send every {constraints.send_every_n_steps} steps")
    
    # Test 2: Frequency constraint
    constraints2 = CommConstraints(
        max_dm=4,
        comm_frequency=0.5,  # Every 2 steps
        topology=TopologyType.FULLY_CONNECTED,
        max_tokens=10
    )
    assert constraints2.send_every_n_steps == 2
    assert constraints2.is_send_allowed_at_step(0) == True
    assert constraints2.is_send_allowed_at_step(1) == False
    assert constraints2.is_send_allowed_at_step(2) == True
    print("Frequency constraint test passed")
    
    # Test 3: Topology
    topology = TopologyManager(
        topology=TopologyType.RING,
        num_agents=4
    )
    assert topology.get_allowed_receivers(0) == [1, 3]
    assert topology.get_allowed_receivers(1) == [2, 0]
    print("Ring topology test passed")
    
    # Test 4: Nearest neighbor
    topology2 = TopologyManager(
        topology=TopologyType.NEAREST_NEIGHBOR,
        num_agents=3,
        comm_range=2,
        grid_size=10,
        agent_positions=[(0, 0), (1, 1), (5, 5)]
    )
    receivers_0 = topology2.get_allowed_receivers(0)
    assert 1 in receivers_0  # Agent 1 is at distance ~1.4 (Chebyshev: 1)
    assert 2 not in receivers_0  # Agent 2 is at distance 5
    print("Nearest neighbor topology test passed")
    
    # Test 5: DecoupledCommBudget integration
    budget = DecoupledCommBudget(
        constraints=CommConstraints(
            max_dm=4,
            comm_frequency=1.0,
            topology=TopologyType.FULLY_CONNECTED,
            max_tokens=5
        ),
        num_agents=2,
        grid_size=10
    )
    
    budget.reset(agent_positions=[(0, 0), (5, 5)])
    
    # Should be able to send initially
    assert budget.can_send(0, 1) == True
    assert budget.can_send(1, 0) == True
    
    # Consume all tokens
    for _ in range(5):
        budget.consume_token(0, 1)
    
    # Should be blocked now
    assert budget.can_send(0, 1) == False
    
    # Test masking
    logits = torch.tensor([0.0, 1.0])
    masked = budget.apply_mask(logits, sender_idx=0)
    assert masked[1] == float('-inf')
    print("DecoupledCommBudget integration test passed")
    
    print("\nAll tests passed!")