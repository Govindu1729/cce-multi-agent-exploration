"""Episode Recorder for Visualization.

Records complete episode trajectories for the Streamlit visualizer.
"""

import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EpisodeStep:
    """Single timestep record."""
    step: int
    grid: np.ndarray              # GxG (0=free, 1=obstacle)
    visited_global: np.ndarray    # GxG (0=unvisited, 1..N=agent_id)
    agent_positions: List[List[int]]  # [[y, x], ...] per agent
    messages: List[Dict[str, Any]]    # [{"sender": int, "receiver": int, "code": int, "utility": float}, ...]
    metrics: Dict[str, float]     # {"coverage": float, "bandwidth_used": float, "lam": float}


class EpisodeRecorder:
    """
    Records episode trajectories for visualization.
    
    Hooks into the evaluation loop to save a complete trajectory.
    """
    
    def __init__(self, env, grid: np.ndarray):
        """
        Initialize recorder.
        
        Args:
            env: VectorGridWorld environment
            grid: Base grid (GxG) with obstacles (1) and free space (0)
        """
        self.env = env
        self.grid = grid
        self.grid_size = grid.shape[0]
        self.num_agents = env.N
        self.trajectory: List[EpisodeStep] = []
        self._episode_step = 0
        self._bandwidth_used = 0.0
        self._lam = 0.0
    
    def reset(self):
        """Reset recorder for new episode."""
        self.trajectory.clear()
        self._episode_step = 0
        self._bandwidth_used = 0.0
        self._lam = 0.0
    
    def record_step(self, obs: Dict, rewards: np.ndarray, info: Dict, 
                    messages: List[Dict[str, Any]]):
        """
        Record a single timestep.
        
        Args:
            obs: Environment observations
            rewards: Rewards for this step
            info: Environment info dict
            messages: List of messages sent this step
        """
        # Build visited_global: 0=unvisited, 1..N=agent_id
        # Use environment's visited_local which is (B, N, G, G)
        visited_global = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i in range(self.num_agents):
            visited_self = self.env.visited_local[0, i]  # (G, G)
            visited_global[visited_self] = i + 1
        
        # Agent positions
        agent_positions = []
        for i in range(self.num_agents):
            y, x = self.env.agent_pos[0, i]
            agent_positions.append([int(y), int(x)])
        
        # Coverage
        coverage = info['coverage'][0]
        
        # Track bandwidth
        self._bandwidth_used += len(messages) * 4  # Assuming dm=4
        
        # Compute lambda (communication efficiency)
        if self._bandwidth_used > 0:
            self._lam = coverage / (self._bandwidth_used + 1e-6)
        
        step_record = EpisodeStep(
            step=self._episode_step,
            grid=self.grid.copy(),
            visited_global=visited_global,
            agent_positions=agent_positions,
            messages=messages.copy(),
            metrics={
                "coverage": float(coverage),
                "bandwidth_used": float(self._bandwidth_used),
                "lam": float(self._lam)
            }
        )
        
        self.trajectory.append(step_record)
        self._episode_step += 1
    
    def save(self, filepath: str):
        """Save trajectory to pickle file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.trajectory, f)
    
    @staticmethod
    def load(filepath: str) -> List[EpisodeStep]:
        """Load trajectory from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_data_for_step(self, step_idx: int) -> Dict[str, Any]:
        """
        Get formatted data for a specific timestep for the visualizer.
        
        Returns:
            Dict with keys: step, grid, visited_global, agent_positions, 
            messages, coverage, bandwidth_used, lam
        """
        if step_idx >= len(self.trajectory):
            step_idx = len(self.trajectory) - 1
        
        record = self.trajectory[step_idx]
        
        return {
            "step": record.step,
            "grid": record.grid,
            "visited_global": record.visited_global,
            "agent_positions": record.agent_positions,
            "messages": record.messages,
            "coverage": record.metrics["coverage"],
            "bandwidth_used": record.metrics["bandwidth_used"],
            "lam": record.metrics["lam"]
        }
    
    def get_coverage_history(self) -> List[float]:
        """Get coverage history over all steps."""
        return [r.metrics["coverage"] for r in self.trajectory]
    
    def get_bandwidth_history(self) -> List[float]:
        """Get bandwidth history over all steps."""
        return [r.metrics["bandwidth_used"] for r in self.trajectory]


def extract_messages_from_info(info: Dict, env) -> List[Dict[str, Any]]:
    """
    Extract messages from environment info.
    
    Expected info['comm_events'] format: [(sender, receiver, mask), ...]
    """
    messages = []
    # VectorGridWorld doesn't track individual messages in info
    # We'll create placeholder messages based on in_range and send flags
    if hasattr(env, 'in_range'):
        in_range = env.in_range[0]  # (N, N)
        for i in range(env.N):
            for j in range(env.N):
                if in_range[i, j] and i != j:
                    messages.append({
                        "sender": i,
                        "receiver": j,
                        "code": 0,
                        "utility": 1.0
                    })
    return messages


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '/Users/srimaan/.openclaw/workspaces/repo-cce')
    
    from cce.envs.grid_world import VectorGridWorld, GridWorldConfig
    
    config = GridWorldConfig(grid_size=20, num_agents=2, seed=42)
    env = VectorGridWorld(config, num_envs=1)
    
    # Create dummy grid
    grid = np.zeros((20, 20), dtype=int)
    grid[5, 5] = 1  # Obstacle
    
    recorder = EpisodeRecorder(env, grid)
    
    # Simulate recording
    obs = env.reset()
    for step in range(5):
        actions = np.array([[0, 1]])  # UP, DOWN
        send_flags = np.array([[False, False]])
        message_embeddings = np.zeros((1, 2, 8), dtype=np.float32)
        obs, rewards, done, info = env.step(actions, send_flags, message_embeddings)
        messages = extract_messages_from_info(info, env)
        recorder.record_step(obs, rewards, info, messages)
        
        if done[0]:
            break
    
    # Save and load
    recorder.save("/tmp/test_trajectory.pkl")
    loaded = EpisodeRecorder.load("/tmp/test_trajectory.pkl")
    
    print(f"Recorded {len(loaded)} steps")
    print("Recorder test passed!")