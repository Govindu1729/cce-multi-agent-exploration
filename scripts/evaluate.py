#!/usr/bin/env python3
"""
Evaluation script for FAI multi-agent coverage experiments.

Supports running baselines (frontier) or trained agents (MAPPO).
Outputs standardized metrics: Coverage, T95, Overlap for comparison.
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/Users/srimaan/.openclaw/workspaces/repo-cce')

from cce.envs.grid_world import GridWorld, GridWorldConfig
from cce.baselines.frontier import (
    FrontierBaseline, 
    FrontierBaselineGlobal, 
    run_frontier_baseline, 
    compute_overlap
)
from cce.eval.metrics import compute_metrics, aggregate_episode_result
from cce.eval.recorder import EpisodeRecorder, extract_messages_from_info


def create_env_from_args(args) -> GridWorld:
    """Create GridWorld environment from command-line arguments."""
    config = GridWorldConfig(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        fov_radius=args.fov_radius,
        max_steps=args.max_steps,
        seed=args.seed,
        max_dm=args.max_dm,
        comm_frequency=args.comm_frequency,
        topology=args.topology,
        comm_budget=args.comm_budget,
        comm_range=args.comm_range,
        tokens_per_send=args.tokens_per_send,
    )
    return GridWorld(config)


def create_grid(env) -> np.ndarray:
    """Create base grid from environment (no obstacles in current env)."""
    grid = np.zeros((env.grid_size, env.grid_size), dtype=int)
    # No obstacles in current GridWorld
    return grid


def run_frontier(args, env) -> dict:
    """Run frontier baseline evaluation."""
    print(f"Running Frontier Baseline...")
    print(f"  Mode: {'Global (upper bound)' if args.global_knowledge else 'Local (decentralized)'}")
    print(f"  Dispersion weight: {args.dispersion_weight}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    
    if args.record:
        # Record a single episode
        return run_frontier_recorded(args, env)
    else:
        metrics = run_frontier_baseline(
            env,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            use_global=args.global_knowledge,
            dispersion_weight=args.dispersion_weight,
            seed=args.seed
        )
        return metrics


def run_frontier_recorded(args, env) -> dict:
    """Run frontier baseline with trajectory recording."""
    # Setup baseline
    if args.global_knowledge:
        baseline = FrontierBaselineGlobal(
            grid_size=env.grid_size,
            num_agents=env.num_agents,
            fov_radius=env.fov_radius,
            dispersion_weight=args.dispersion_weight
        )
    else:
        baseline = FrontierBaseline(
            grid_size=env.grid_size,
            num_agents=env.num_agents,
            fov_radius=env.fov_radius,
            dispersion_weight=args.dispersion_weight,
            comm_range=env.comm_range
        )
    
    # Setup recorder
    grid = create_grid(env)
    recorder = EpisodeRecorder(env, grid)
    
    # Run single episode
    obs = env.reset()
    recorder.reset()
    
    # Get initial global state for global baseline
    if args.global_knowledge:
        global_state = env._get_global_state()
        global_visited = global_state['global_visited'].numpy()
    else:
        global_visited = np.zeros((env.grid_size, env.grid_size), dtype=bool)
        for i in range(env.num_agents):
            global_visited |= obs['visited_self'][i].numpy()
    
    for step in range(args.max_steps):
        # Get observations
        if args.global_knowledge:
            global_state = env._get_global_state()
            global_visited = global_state['global_visited'].numpy()
        
        # Compute actions
        if args.global_knowledge:
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
        
        # Extract messages
        messages = extract_messages_from_info(info)
        
        # Record step
        recorder.record_step(obs, rewards, info, messages)
        
        if done:
            break
    
    # Save trajectory
    output_path = Path(args.record_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recorder.save(str(output_path))
    print(f"Trajectory saved to {output_path} ({len(recorder.trajectory)} steps)")
    
    # Return final metrics
    final_coverage = info['global_coverage']
    overlap = compute_overlap(env.visited_masks)
    
    return {
        'coverage_mean': final_coverage,
        'coverage_std': 0.0,
        'coverage_final': final_coverage,
        't95': float('inf'),
        't95_mean': float('inf'),
        't95_std': 0.0,
        'success_rate': 1.0 if final_coverage >= 0.95 else 0.0,
        'overlap_mean': overlap,
        'overlap_std': 0.0,
        'steps_mean': len(recorder.trajectory),
        'steps_std': 0.0,
    }


def run_random_baseline(args, env) -> dict:
    """Run random action baseline for comparison."""
    print(f"Running Random Baseline...")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    
    np.random.seed(args.seed)
    
    all_coverages = []
    all_steps_to_95 = []
    all_overlaps = []
    all_episode_steps = []
    
    for ep in range(args.num_episodes):
        obs = env.reset()
        coverage_reached_95 = False
        steps_to_95 = args.max_steps
        
        for step in range(args.max_steps):
            # Random actions (movement only, no SEND)
            actions = [np.random.randint(0, 5) for _ in range(env.num_agents)]
            
            obs, rewards, done, info = env.step(actions)
            coverage = info['global_coverage']
            
            if not coverage_reached_95 and coverage >= 0.95:
                coverage_reached_95 = True
                steps_to_95 = step + 1
            
            if done:
                break
        
        final_coverage = info['global_coverage']
        all_coverages.append(final_coverage)
        all_episode_steps.append(step + 1)
        
        if coverage_reached_95:
            all_steps_to_95.append(steps_to_95)
        
        overlap = compute_overlap(env.visited_masks)
        all_overlaps.append(overlap)
    
    metrics = {
        'coverage_mean': np.mean(all_coverages),
        'coverage_std': np.std(all_coverages),
        'coverage_final': all_coverages[-1] if all_coverages else 0.0,
        't95': np.median(all_steps_to_95) if all_steps_to_95 else float('inf'),
        't95_mean': np.mean(all_steps_to_95) if all_steps_to_95 else float('inf'),
        't95_std': np.std(all_steps_to_95) if all_steps_to_95 else 0.0,
        'success_rate': len(all_steps_to_95) / args.num_episodes,
        'overlap_mean': np.mean(all_overlaps),
        'overlap_std': np.std(all_overlaps),
        'steps_mean': np.mean(all_episode_steps),
        'steps_std': np.std(all_episode_steps),
    }
    
    return metrics


def print_metrics(metrics: dict, label: str = "Metrics"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"{'Metric':<25} {'Value':>15}")
    print(f"{'-'*25} {'-'*15}")
    
    # Primary metrics for paper comparison
    primary = ['coverage_mean', 'coverage_std', 't95', 't95_std', 
               'success_rate', 'overlap_mean', 'overlap_std']
    
    for key in primary:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                if val == float('inf'):
                    print(f"{key:<25} {'inf':>15}")
                else:
                    print(f"{key:<25} {val:>15.4f}")
            else:
                print(f"{key:<25} {val:>15}")
    
    print(f"{'-'*25} {'-'*15}")
    
    # Secondary metrics
    secondary = ['coverage_final', 't95_mean', 'steps_mean', 'steps_std']
    for key in secondary:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                if val == float('inf'):
                    print(f"{key:<25} {'inf':>15}")
                else:
                    print(f"{key:<25} {val:>15.4f}")
            else:
                print(f"{key:<25} {val:>15}")
    
    print(f"{'='*50}")


def save_metrics(metrics: dict, output_path: Path):
    """Save metrics to JSON file."""
    import json
    
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64)):
            serializable[k] = float(v) if v != float('inf') else "inf"
        elif isinstance(v, (np.int32, np.int64)):
            serializable[k] = int(v)
        else:
            serializable[k] = v
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nMetrics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FAI multi-agent coverage baselines and agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run frontier baseline (decentralized)
  python evaluate.py --baseline frontier --num-episodes 100
  
  # Run frontier baseline with global knowledge (upper bound)
  python evaluate.py --baseline frontier --global-knowledge --num-episodes 100
  
  # Run random baseline
  python evaluate.py --baseline random --num-episodes 100
  
  # Record trajectory for visualization
  python evaluate.py --baseline frontier --record --record-output runs/viz/trajectory.pkl
  
  # Run with custom communication constraints
  python evaluate.py --baseline frontier --comm-frequency 0.5 --topology ring
        """
    )
    
    # Baseline selection
    parser.add_argument(
        '--baseline', 
        type=str, 
        choices=['frontier', 'random'],
        default='frontier',
        help='Baseline to evaluate (default: frontier)'
    )
    
    # Frontier-specific options
    parser.add_argument(
        '--global-knowledge', 
        action='store_true',
        help='Use global knowledge for frontier (upper bound)'
    )
    parser.add_argument(
        '--dispersion-weight', 
        type=float, 
        default=1.0,
        help='Dispersion heuristic weight for multi-agent (default: 1.0)'
    )
    
    # Recording options
    parser.add_argument(
        '--record', 
        action='store_true',
        help='Record trajectory for visualization (single episode)'
    )
    parser.add_argument(
        '--record-output', 
        type=str, 
        default='runs/viz/trajectory.pkl',
        help='Output path for recorded trajectory (default: runs/viz/trajectory.pkl)'
    )
    
    # Environment configuration
    parser.add_argument('--grid-size', type=int, default=10, help='Grid size (default: 10)')
    parser.add_argument('--num-agents', type=int, default=4, help='Number of agents (default: 4)')
    parser.add_argument('--fov-radius', type=int, default=2, help='Field of view radius (default: 2)')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per episode (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    # Communication constraints (decoupled)
    parser.add_argument('--max-dm', type=int, default=4, help='Message dimension in bits (default: 4)')
    parser.add_argument('--comm-frequency', type=float, default=1.0, 
                       help='Communication frequency (0<f<=1, default: 1.0)')
    parser.add_argument('--topology', type=str, default='fully_connected',
                       choices=['fully_connected', 'nearest_neighbor', 'ring', 'star', 'random'],
                       help='Communication topology (default: fully_connected)')
    parser.add_argument('--comm-budget', type=int, default=10, help='Token budget (default: 10)')
    parser.add_argument('--comm-range', type=int, default=3, help='Comm range for nearest_neighbor (default: 3)')
    parser.add_argument('--tokens-per-send', type=int, default=1, help='Tokens per send (default: 1)')
    
    # Evaluation configuration
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes (default: 100)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = create_env_from_args(args)
    
    print(f"Environment: {args.grid_size}x{args.grid_size} grid, {args.num_agents} agents, "
          f"FOV radius {args.fov_radius}, {args.max_steps} steps")
    print(f"Comm: dm={args.max_dm}, freq={args.comm_frequency}, topo={args.topology}, "
          f"budget={args.comm_budget}, range={args.comm_range}")
    
    if args.record:
        print(f"Recording trajectory to {args.record_output}")
    
    # Run selected baseline
    if args.baseline == 'frontier':
        metrics = run_frontier(args, env)
        label = f"Frontier Baseline {'(Global)' if args.global_knowledge else '(Decentralized)'}"
    elif args.baseline == 'random':
        metrics = run_random_baseline(args, env)
        label = "Random Baseline"
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")
    
    # Print and save metrics
    print_metrics(metrics, label)
    
    if args.output:
        save_metrics(metrics, Path(args.output))
    
    # Return exit code based on success
    if metrics.get('success_rate', 0) > 0:
        return 0
    else:
        print("WARNING: Zero success rate!")
        return 1


if __name__ == "__main__":
    sys.exit(main())