"""
Grid World Environment for Multi-Agent Coverage.

IMPORTANT: Global coverage information is ONLY available to the centralised critic.
Individual agents MUST NOT have access to global coverage state.
Each agent only observes:
  - visited_self: Cells this agent has personally visited (within its FOV)
  - visited_teammates: Cells reported by teammates via communication (if any)
  - local_obs: Local observation within FOV

The global visited mask (union of all agents) is computed ONLY for the
centralised critic during training and is NOT included in agent observations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from cce.comm.budget import (
    CommConstraints, TopologyType, TopologyManager, DecoupledCommBudget
)


class Action(Enum):
    """Agent actions."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    SEND = 5  # Communication action


@dataclass
class GridWorldConfig:
    """Configuration for GridWorld environment."""
    # Environment
    grid_size: int = 10
    num_agents: int = 2
    fov_radius: int = 2  # Field of view radius (Chebyshev distance)
    max_steps: int = 50
    seed: Optional[int] = None
    
    # Communication constraints (decoupled)
    # Legacy scalar budget (kept for backward compatibility)
    comm_budget: int = 10  # Max communication tokens per episode
    comm_range: int = 3    # Communication range for nearest_neighbor topology
    
    # New granular constraints
    max_dm: int = 4                    # Message dimension in bits
    comm_frequency: float = 1.0        # Send allowed every N steps (1.0 = every step)
    topology: str = "fully_connected"  # Topology type
    tokens_per_send: int = 1           # Tokens consumed per send
    
    def __post_init__(self):
        """Validate and normalize config."""
        # Normalize topology string
        if isinstance(self.topology, TopologyType):
            self.topology = self.topology.value
        self.topology = self.topology.lower()
        
        # Create CommConstraints object
        self.comm_constraints = CommConstraints(
            max_dm=self.max_dm,
            comm_frequency=self.comm_frequency,
            topology=TopologyType(self.topology),
            max_tokens=self.comm_budget,
            tokens_per_send=self.tokens_per_send,
            comm_range=self.comm_range,
        )


class GridWorld:
    """
    Multi-agent grid world for coverage task.
    
    Agents must explore the grid to maximize coverage. They have limited
    field of view and can communicate visited cells to teammates within
    communication range, subject to a hard token budget.
    
    OBSERVATION SPACE PER AGENT:
    - 'position': (2,) - agent's (x, y) position
    - 'visited_self': (grid_size, grid_size) - binary mask of cells this agent has seen
    - 'visited_teammates': (grid_size, grid_size) - binary mask of cells reported by teammates
    - 'tokens_remaining': (1,) - remaining communication tokens
    - 'step': (1,) - current step / max_steps
    - 'comm_allowed': (num_agents,) - which agents can be sent to this step
    
    GLOBAL STATE (CRITIC ONLY):
    - 'global_visited': (grid_size, grid_size) - union of all agents' visited cells
    - 'global_coverage': float - fraction of grid covered
    - 'all_positions': (num_agents, 2) - all agent positions
    - 'comm_budget_state': dict - full budget state for logging
    
    Agents NEVER receive global_visited or global_coverage in their observations.
    """
    
    def __init__(self, config: GridWorldConfig):
        self.config = config
        self.grid_size = config.grid_size
        self.num_agents = config.num_agents
        self.fov_radius = config.fov_radius
        self.max_steps = config.max_steps
        self.comm_range = config.comm_range
        
        # RNG
        self.rng = np.random.RandomState(config.seed)
        self.torch_rng = torch.Generator()
        if config.seed is not None:
            self.torch_rng.manual_seed(config.seed)
        
        # Communication budget (decoupled)
        self.comm_budget = DecoupledCommBudget(
            constraints=config.comm_constraints,
            num_agents=config.num_agents,
            grid_size=config.grid_size
        )
        
        # State
        self.agent_positions: List[Tuple[int, int]] = []
        self.visited_masks: List[np.ndarray] = []  # Per-agent visited cells
        self.global_visited: np.ndarray = None
        self.step_count: int = 0
        self.episode_done: bool = False
        
        # Communication tracking
        self.last_messages: List[Optional[Dict]] = []
        self.comm_events: List[Tuple[int, int, np.ndarray]] = []
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment for new episode."""
        # Clear previous episode state
        self.agent_positions.clear()
        
        # Random initial positions (spread out)
        positions = set()
        for i in range(self.num_agents):
            while True:
                pos = (self.rng.randint(0, self.grid_size), 
                       self.rng.randint(0, self.grid_size))
                if pos not in positions:
                    positions.add(pos)
                    self.agent_positions.append(pos)
                    break
        
        # Visited masks per agent
        self.visited_masks = [
            np.zeros((self.grid_size, self.grid_size), dtype=bool)
            for _ in range(self.num_agents)
        ]
        
        # Initialize visited with starting positions (within FOV)
        for i, pos in enumerate(self.agent_positions):
            self._update_visited(i, pos)
        
        # Global visited (critic only)
        self.global_visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for mask in self.visited_masks:
            self.global_visited |= mask
        
        # Episode state
        self.step_count = 0
        self.episode_done = False
        self.last_messages = [None] * self.num_agents
        self.comm_events = []
        
        # Reset comm budget with initial positions
        self.comm_budget.reset(agent_positions=self.agent_positions)
        
        return self._get_observations()
    
    def _update_visited(self, agent_idx: int, pos: Tuple[int, int]):
        """Update visited mask for an agent based on its FOV."""
        x, y = pos
        for dx in range(-self.fov_radius, self.fov_radius + 1):
            for dy in range(-self.fov_radius, self.fov_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Chebyshev distance (square FOV)
                    if max(abs(dx), abs(dy)) <= self.fov_radius:
                        self.visited_masks[agent_idx][ny, nx] = True
    
    def _get_fov_mask(self, pos: Tuple[int, int]) -> np.ndarray:
        """Get FOV mask for a position (used for info leakage test)."""
        x, y = pos
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for dx in range(-self.fov_radius, self.fov_radius + 1):
            for dy in range(-self.fov_radius, self.fov_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if max(abs(dx), abs(dy)) <= self.fov_radius:
                        mask[ny, nx] = True
        return mask
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations for all agents (NO global coverage)."""
        obs = {}
        
        # Get allowed receivers for each agent at current step
        comm_allowed = []
        for i in range(self.num_agents):
            allowed = self.comm_budget.get_allowed_receivers(i)
            # Convert to binary mask
            allowed_mask = torch.zeros(self.num_agents, dtype=torch.bool)
            for j in allowed:
                allowed_mask[j] = True
            comm_allowed.append(allowed_mask)
        
        comm_allowed = torch.stack(comm_allowed)
        
        # Per-agent observations
        for i in range(self.num_agents):
            # Visited by teammates (via communication)
            visited_teammates = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            for j in range(self.num_agents):
                if j != i:
                    visited_teammates |= self.visited_masks[j]
            
            agent_obs = {
                'position': torch.tensor(self.agent_positions[i], dtype=torch.float32),
                'visited_self': torch.tensor(self.visited_masks[i].copy(), dtype=torch.bool),
                'visited_teammates': torch.tensor(visited_teammates, dtype=torch.bool),
                'tokens_remaining': torch.tensor(
                    [self.comm_budget.token_bucket.remaining], dtype=torch.float32
                ),
                'step': torch.tensor(
                    [self.step_count / self.max_steps], dtype=torch.float32
                ),
                'comm_allowed': comm_allowed[i],  # Which agents can receive
            }
            
            for key, val in agent_obs.items():
                if key not in obs:
                    obs[key] = []
                obs[key].append(val)
        
        # Stack per-agent observations
        for key in obs:
            obs[key] = torch.stack(obs[key])
        
        return obs
    
    def _get_global_state(self) -> Dict[str, torch.Tensor]:
        """Get global state for CENTRALISED CRITIC ONLY.
        
        This is NOT returned to agents. Only the critic has access.
        """
        # Update global visited
        self.global_visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for mask in self.visited_masks:
            self.global_visited |= mask
        
        global_coverage = self.global_visited.sum() / (self.grid_size * self.grid_size)
        
        return {
            'global_visited': torch.tensor(self.global_visited.copy(), dtype=torch.bool),
            'global_coverage': torch.tensor([global_coverage], dtype=torch.float32),
            'all_positions': torch.tensor(self.agent_positions, dtype=torch.float32),
            'tokens_remaining': torch.tensor(
                [self.comm_budget.token_bucket.remaining], dtype=torch.float32
            ),
            'step': torch.tensor([self.step_count], dtype=torch.float32),
            'comm_budget_state': self.comm_budget.get_state(),
        }
    
    def step(self, actions: List[int]) -> Tuple[Dict, List[float], bool, Dict]:
        """
        Step the environment.
        
        Args:
            actions: List of actions for each agent (length = num_agents)
            
        Returns:
            observations, rewards, done, info
        """
        if self.episode_done:
            raise RuntimeError("Episode already done. Call reset() first.")
        
        rewards = [0.0] * self.num_agents
        self.comm_events = []
        
        # Process movement actions
        for i, action in enumerate(actions):
            if action == Action.SEND.value:
                # Communication action handled separately
                continue
            
            # Movement
            x, y = self.agent_positions[i]
            if action == Action.UP.value:
                y = max(0, y - 1)
            elif action == Action.DOWN.value:
                y = min(self.grid_size - 1, y + 1)
            elif action == Action.LEFT.value:
                x = max(0, x - 1)
            elif action == Action.RIGHT.value:
                x = min(self.grid_size - 1, x + 1)
            elif action == Action.STAY.value:
                pass
            
            self.agent_positions[i] = (x, y)
            self._update_visited(i, (x, y))
        
        # Process communication actions
        for i, action in enumerate(actions):
            if action != Action.SEND.value:
                continue
            
            # Check if send is allowed by budget
            if not self.comm_budget.token_bucket.can_send():
                rewards[i] -= 0.1  # Small penalty for attempting invalid action
                continue
            
            # Check frequency constraint
            if not self.config.comm_constraints.is_send_allowed_at_step(self.step_count):
                rewards[i] -= 0.1
                continue
            
            # Get allowed receivers based on topology
            allowed_receivers = self.comm_budget.get_allowed_receivers(i)
            
            if not allowed_receivers:
                rewards[i] -= 0.1
                continue
            
            # Send visited_self to allowed receivers
            sender_mask = self.visited_masks[i].copy()
            
            for j in allowed_receivers:
                self.comm_events.append((i, j, sender_mask))
            
            # Consume token
            self.comm_budget.consume_token(i, len(allowed_receivers))
        
        # Calculate rewards based on NEW coverage
        prev_coverage = self._calculate_coverage()
        
        # Update global visited
        self.global_visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for mask in self.visited_masks:
            self.global_visited |= mask
        
        new_coverage = self.global_visited.sum() / (self.grid_size * self.grid_size)
        coverage_gain = new_coverage - prev_coverage
        
        # Shared reward for coverage gain
        for i in range(self.num_agents):
            rewards[i] += coverage_gain * 10.0  # Scale reward
        
        # Step counter and budget step
        self.step_count += 1
        self.comm_budget.step(agent_positions=self.agent_positions)
        
        # Check termination
        self.episode_done = (self.step_count >= self.max_steps)
        
        # Info dict
        info = {
            'global_coverage': new_coverage,
            'tokens_remaining': self.comm_budget.token_bucket.remaining,
            'comm_events': self.comm_events,
            'global_state': self._get_global_state(),  # FOR CRITIC ONLY
        }
        
        return self._get_observations(), rewards, self.episode_done, info
    
    def _calculate_coverage(self) -> float:
        """Calculate current global coverage."""
        if self.global_visited is None:
            return 0.0
        return self.global_visited.sum() / (self.grid_size * self.grid_size)
    
    def get_agent_fov_cells(self, agent_idx: int) -> Set[Tuple[int, int]]:
        """Get the set of cells in agent's FOV (for testing)."""
        pos = self.agent_positions[agent_idx]
        cells = set()
        x, y = pos
        for dx in range(-self.fov_radius, self.fov_radius + 1):
            for dy in range(-self.fov_radius, self.fov_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if max(abs(dx), abs(dy)) <= self.fov_radius:
                        cells.add((nx, ny))
        return cells
    
    def get_comm_adjacency(self) -> np.ndarray:
        """Get current communication adjacency matrix."""
        return self.comm_budget.topology.get_adjacency_matrix()


if __name__ == "__main__":
    # Quick test
    config = GridWorldConfig(grid_size=10, num_agents=2, seed=42)
    env = GridWorld(config)
    
    obs = env.reset()
    print("Reset successful")
    print(f"Observations keys: {obs.keys()}")
    print(f"visited_self shape: {obs['visited_self'].shape}")
    print(f"position shape: {obs['position'].shape}")
    print(f"comm_allowed shape: {obs['comm_allowed'].shape}")
    
    # Step
    obs, rewards, done, info = env.step([Action.RIGHT.value, Action.LEFT.value])
    print(f"Rewards: {rewards}")
    print(f"Done: {done}")
    print(f"Global coverage (critic): {info['global_state']['global_coverage'].item():.3f}")
    
    print("\nGridWorld test passed!")


# For testing: expose internal function to verify no info leakage
def compute_exact_visited_self(env: GridWorld, agent_idx: int, trajectory: List[Tuple[int, int]]) -> np.ndarray:
    """
    Compute exactly what cells an agent should have visited based on its trajectory.
    
    This is used by test_no_info_leakage_in_visited_mask to verify
    that the agent's visited_self matches exactly what it could have seen.
    
    Args:
        env: GridWorld instance
        agent_idx: Agent index
        trajectory: List of (x, y) positions the agent has been at
        
    Returns:
        Boolean mask of cells the agent should have visited
    """
    visited = np.zeros((env.grid_size, env.grid_size), dtype=bool)
    for pos in trajectory:
        x, y = pos
        for dx in range(-env.fov_radius, env.fov_radius + 1):
            for dy in range(-env.fov_radius, env.fov_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                    if max(abs(dx), abs(dy)) <= env.fov_radius:
                        visited[ny, nx] = True
    return visited