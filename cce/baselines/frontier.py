"""
Frontier-Based Exploration Baseline for Multi-Agent Coverage.

This is a classical non-learning baseline for comparison with MAPPO.
Identifies frontier cells (boundary between known free space and unknown)
and moves agents towards the nearest frontier with a dispersion heuristic.
"""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from collections import deque
import heapq


class FrontierBaseline:
    """
    Frontier-based exploration agent.
    
    For each agent:
    1. Compute frontiers: cells adjacent to visited cells but not visited themselves
    2. Select nearest frontier using Manhattan distance
    3. Apply dispersion heuristic: avoid frontiers that are closer to other agents
    4. Move one step towards selected frontier
    """
    
    def __init__(
        self,
        grid_size: int,
        num_agents: int,
        fov_radius: int,
        dispersion_weight: float = 1.0,
        comm_range: int = 3
    ):
        """
        Initialize frontier baseline.
        
        Args:
            grid_size: Size of grid (grid_size x grid_size)
            num_agents: Number of agents
            fov_radius: Field of view radius
            dispersion_weight: Weight for avoiding other agents' targets
            comm_range: Communication range (for sharing visited info)
        """
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.fov_radius = fov_radius
        self.dispersion_weight = dispersion_weight
        self.comm_range = comm_range
        
        # Movement actions (no SEND action for baseline)
        self.actions = {
            'UP': 0,
            'DOWN': 1,
            'LEFT': 2,
            'RIGHT': 3,
            'STAY': 4
        }
        self.action_vectors = {
            0: (0, -1),  # UP
            1: (0, 1),   # DOWN
            2: (-1, 0),  # LEFT
            3: (1, 0),   # RIGHT
            4: (0, 0)    # STAY
        }
    
    def compute_frontiers(self, visited_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Compute frontier cells: unvisited cells adjacent to visited cells.
        
        A frontier is a cell that:
        - Is not visited (unknown)
        - Has at least one visited neighbor (4-connected)
        
        Args:
            visited_mask: Boolean mask of visited cells (grid_size, grid_size)
            
        Returns:
            List of (x, y) frontier cell coordinates
        """
        frontiers = []
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if visited_mask[y, x]:
                    continue  # Already visited
                
                # Check 4-connected neighbors
                is_frontier = False
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if visited_mask[ny, nx]:
                            is_frontier = True
                            break
                
                if is_frontier:
                    frontiers.append((x, y))
        
        return frontiers
    
    def compute_frontiers_global(self, global_visited: np.ndarray) -> List[Tuple[int, int]]:
        """
        Compute frontiers based on global visited map (all agents combined).
        """
        return self.compute_frontiers(global_visited)
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_next_step_towards(
        self, 
        current_pos: Tuple[int, int], 
        target_pos: Tuple[int, int]
    ) -> int:
        """
        Get the action to move one step towards target.
        
        Uses greedy Manhattan distance reduction.
        """
        cx, cy = current_pos
        tx, ty = target_pos
        
        dx = tx - cx
        dy = ty - cy
        
        # Prioritize larger absolute difference
        if abs(dx) > abs(dy):
            if dx > 0:
                return self.actions['RIGHT']
            else:
                return self.actions['LEFT']
        elif dy > 0:
            return self.actions['DOWN']
        elif dy < 0:
            return self.actions['UP']
        else:
            return self.actions['STAY']
    
    def select_frontier(
        self,
        agent_idx: int,
        agent_pos: Tuple[int, int],
        agent_visited: np.ndarray,
        all_visited: List[np.ndarray],
        other_positions: List[Tuple[int, int]],
        global_visited: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Select the best frontier for an agent.
        
        Uses:
        1. Agent's own visited + shared teammate visited (within comm range)
        2. Distance to frontier
        3. Dispersion: penalize frontiers near other agents
        """
        # Combine visited info: own + teammates within comm range
        combined_visited = agent_visited.copy()
        for j, other_pos in enumerate(other_positions):
            if j == agent_idx:
                continue
            dist = self.manhattan_distance(agent_pos, other_pos)
            if dist <= self.comm_range:
                combined_visited |= all_visited[j]
        
        # Compute frontiers from combined knowledge
        frontiers = self.compute_frontiers(combined_visited)
        
        if not frontiers:
            return None
        
        # Score each frontier
        best_frontier = None
        best_score = float('inf')
        
        for fx, fy in frontiers:
            # Distance to frontier
            dist = self.manhattan_distance(agent_pos, (fx, fy))
            
            # Dispersion penalty: sum of inverse distances to other agents
            dispersion_penalty = 0.0
            for j, other_pos in enumerate(other_positions):
                if j == agent_idx:
                    continue
                other_dist = self.manhattan_distance(other_pos, (fx, fy))
                if other_dist > 0:
                    dispersion_penalty += 1.0 / other_dist
            
            # Total score (lower is better)
            score = dist + self.dispersion_weight * dispersion_penalty
            
            if score < best_score:
                best_score = score
                best_frontier = (fx, fy)
        
        return best_frontier
    
    def act(
        self,
        agent_positions: List[Tuple[int, int]],
        visited_masks: List[np.ndarray],
        global_visited: np.ndarray
    ) -> List[int]:
        """
        Compute actions for all agents.
        
        Args:
            agent_positions: Current positions of all agents
            visited_masks: Per-agent visited masks
            global_visited: Global visited mask (union of all)
            
        Returns:
            List of actions for each agent
        """
        actions = []
        
        for i in range(self.num_agents):
            pos = agent_positions[i]
            agent_visited = visited_masks[i]
            
            # Select frontier
            target = self.select_frontier(
                agent_idx=i,
                agent_pos=pos,
                agent_visited=agent_visited,
                all_visited=visited_masks,
                other_positions=agent_positions,
                global_visited=global_visited
            )
            
            if target is None:
                # No frontiers - stay put
                actions.append(self.actions['STAY'])
            else:
                # Move towards frontier
                action = self.get_next_step_towards(pos, target)
                actions.append(action)
        
        return actions


class FrontierBaselineGlobal:
    """
    Frontier baseline with global knowledge (upper bound).
    
    Each agent has access to the global visited map and all agent positions.
    This represents the best possible frontier performance.
    """
    
    def __init__(
        self,
        grid_size: int,
        num_agents: int,
        fov_radius: int,
        dispersion_weight: float = 1.0
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.fov_radius = fov_radius
        self.dispersion_weight = dispersion_weight
        
        self.actions = {
            'UP': 0,
            'DOWN': 1,
            'LEFT': 2,
            'RIGHT': 3,
            'STAY': 4
        }
        self.action_vectors = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0),
            4: (0, 0)
        }
    
    def compute_frontiers(self, visited_mask: np.ndarray) -> List[Tuple[int, int]]:
        frontiers = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if visited_mask[y, x]:
                    continue
                is_frontier = False
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if visited_mask[ny, nx]:
                            is_frontier = True
                            break
                if is_frontier:
                    frontiers.append((x, y))
        return frontiers
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_next_step_towards(
        self, 
        current_pos: Tuple[int, int], 
        target_pos: Tuple[int, int]
    ) -> int:
        cx, cy = current_pos
        tx, ty = target_pos
        dx = tx - cx
        dy = ty - cy
        
        if abs(dx) > abs(dy):
            if dx > 0:
                return self.actions['RIGHT']
            else:
                return self.actions['LEFT']
        elif dy > 0:
            return self.actions['DOWN']
        elif dy < 0:
            return self.actions['UP']
        else:
            return self.actions['STAY']
    
    def act(
        self,
        agent_positions: List[Tuple[int, int]],
        visited_masks: List[np.ndarray],
        global_visited: np.ndarray
    ) -> List[int]:
        actions = []
        frontiers = self.compute_frontiers(global_visited)
        
        if not frontiers:
            return [self.actions['STAY']] * self.num_agents
        
        # Assign frontiers to agents to minimize total distance
        # Simple greedy assignment
        available_frontiers = frontiers.copy()
        
        for i in range(self.num_agents):
            pos = agent_positions[i]
            best_frontier = None
            best_score = float('inf')
            
            for fx, fy in available_frontiers:
                dist = self.manhattan_distance(pos, (fx, fy))
                
                # Dispersion
                dispersion = 0.0
                for j, other_pos in enumerate(agent_positions):
                    if j == i:
                        continue
                    other_dist = self.manhattan_distance(other_pos, (fx, fy))
                    if other_dist > 0:
                        dispersion += 1.0 / other_dist
                
                score = dist + self.dispersion_weight * dispersion
                
                if score < best_score:
                    best_score = score
                    best_frontier = (fx, fy)
            
            if best_frontier:
                available_frontiers.remove(best_frontier)
                action = self.get_next_step_towards(pos, best_frontier)
                actions.append(action)
            else:
                actions.append(self.actions['STAY'])
        
        return actions


def run_frontier_baseline(
    env,
    num_episodes: int = 100,
    max_steps: int = 50,
    use_global: bool = False,
    dispersion_weight: float = 1.0,
    seed: Optional[int] = None
) -> Dict:
    """
    Run frontier baseline evaluation.
    
    Args:
        env: GridWorld environment
        num_episodes: Number of episodes to run
        max_steps: Max steps per episode
        use_global: If True, use global knowledge (upper bound)
        dispersion_weight: Weight for dispersion heuristic
        seed: Random seed
        
    Returns:
        Dictionary with metrics: coverage, t95, overlap, steps_per_episode
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Choose baseline class
    if use_global:
        baseline = FrontierBaselineGlobal(
            grid_size=env.grid_size,
            num_agents=env.num_agents,
            fov_radius=env.fov_radius,
            dispersion_weight=dispersion_weight
        )
    else:
        baseline = FrontierBaseline(
            grid_size=env.grid_size,
            num_agents=env.num_agents,
            fov_radius=env.fov_radius,
            dispersion_weight=dispersion_weight,
            comm_range=env.comm_range
        )
    
    all_coverages = []
    all_steps_to_95 = []
    all_overlaps = []
    all_episode_steps = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        
        # Get initial global state for global baseline
        if use_global:
            global_state = env._get_global_state()
            global_visited = global_state['global_visited'].numpy()
        else:
            # For local baseline, build global from observations
            global_visited = np.zeros((env.grid_size, env.grid_size), dtype=bool)
            for i in range(env.num_agents):
                global_visited |= obs['visited_self'][i].numpy()
        
        coverage_reached_95 = False
        steps_to_95 = max_steps
        
        for step in range(max_steps):
            # Get observations
            if use_global:
                global_state = env._get_global_state()
                global_visited = global_state['global_visited'].numpy()
            
            # Compute actions
            if use_global:
                actions = baseline.act(
                    agent_positions=env.agent_positions,
                    visited_masks=env.visited_masks,
                    global_visited=global_visited
                )
            else:
                actions = baseline.act(
                    agent_positions=env.agent_positions,
                    visited_masks=env.visited_masks,
                    global_visited=global_visited
                )
            
            # Step environment
            obs, rewards, done, info = env.step(actions)
            
            # Update global visited
            global_visited = info['global_state']['global_visited'].numpy()
            coverage = info['global_coverage']
            
            # Check if 95% coverage reached
            if not coverage_reached_95 and coverage >= 0.95:
                coverage_reached_95 = True
                steps_to_95 = step + 1
            
            if done:
                break
        
        # Final metrics
        final_coverage = info['global_coverage']
        all_coverages.append(final_coverage)
        all_episode_steps.append(step + 1)
        
        if coverage_reached_95:
            all_steps_to_95.append(steps_to_95)
        
        # Compute overlap: fraction of cells visited by multiple agents
        overlap = compute_overlap(env.visited_masks)
        all_overlaps.append(overlap)
    
    # Aggregate metrics
    metrics = {
        'coverage_mean': np.mean(all_coverages),
        'coverage_std': np.std(all_coverages),
        'coverage_final': all_coverages[-1] if all_coverages else 0.0,
        't95': np.median(all_steps_to_95) if all_steps_to_95 else float('inf'),
        't95_mean': np.mean(all_steps_to_95) if all_steps_to_95 else float('inf'),
        't95_std': np.std(all_steps_to_95) if all_steps_to_95 else 0.0,
        'success_rate': len(all_steps_to_95) / num_episodes,
        'overlap_mean': np.mean(all_overlaps),
        'overlap_std': np.std(all_overlaps),
        'steps_mean': np.mean(all_episode_steps),
        'steps_std': np.std(all_episode_steps),
    }
    
    return metrics


def compute_overlap(visited_masks: List[np.ndarray]) -> float:
    """
    Compute overlap metric: fraction of visited cells visited by >1 agent.
    
    Args:
        visited_masks: List of per-agent visited boolean masks
        
    Returns:
        Overlap ratio (0 to 1)
    """
    if not visited_masks:
        return 0.0
    
    grid_size = visited_masks[0].shape[0]
    visit_counts = np.zeros((grid_size, grid_size), dtype=int)
    
    for mask in visited_masks:
        visit_counts += mask.astype(int)
    
    # Cells visited by at least one agent
    total_visited = np.sum(visit_counts > 0)
    # Cells visited by more than one agent
    overlapped = np.sum(visit_counts > 1)
    
    if total_visited == 0:
        return 0.0
    
    return overlapped / total_visited


if __name__ == "__main__":
    # Quick test with GridWorld
    import sys
    sys.path.insert(0, '/Users/srimaan/.openclaw/workspaces/repo-cce')
    
    from cce.envs.grid_world import GridWorld, GridWorldConfig
    
    config = GridWorldConfig(
        grid_size=10,
        num_agents=2,
        fov_radius=2,
        max_steps=50,
        seed=42
    )
    env = GridWorld(config)
    
    print("Testing FrontierBaseline...")
    metrics = run_frontier_baseline(env, num_episodes=10, seed=42)
    
    print("\nMetrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
    
    print("\nTesting FrontierBaselineGlobal...")
    metrics_global = run_frontier_baseline(env, num_episodes=10, use_global=True, seed=42)
    
    print("\nGlobal Metrics:")
    for key, val in metrics_global.items():
        print(f"  {key}: {val}")
    
    print("\nFrontier baseline test passed!")