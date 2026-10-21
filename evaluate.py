#!/usr/bin/env python3
"""Evaluate a trained model or baseline over multiple episodes and compute final metrics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cce.envs.grid_world import VectorGridWorld, GridWorldConfig
from cce.algo.mappo import MAPPO
from cce.eval.metrics import compute_episode_metrics
from cce.eval.protocol import message_statistics
from cce.utils.seeding import set_seed
from cce.eval.recorder import EpisodeRecorder, extract_messages_from_info
from cce.baselines.frontier import run_frontier_baseline, run_random_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO or baselines")
    parser.add_argument("--run_dir", type=str, default=None,
                       help="Path to training run directory (for MAPPO evaluation)")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt",
                       help="Checkpoint file name")
    parser.add_argument("--baseline", type=str, choices=['frontier', 'random', None],
                       default=None,
                       help="Run baseline instead of MAPPO: 'frontier' or 'random'")
    parser.add_argument("--global-knowledge", action="store_true",
                       help="Use global knowledge for frontier baseline (upper bound)")
    parser.add_argument("--dispersion-weight", type=float, default=1.0,
                       help="Dispersion heuristic weight for frontier baseline")
    parser.add_argument("--num_episodes", type=int, default=50,
                       help="Number of episodes to evaluate")
    parser.add_argument("--num_envs", type=int, default=16,
                       help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", type=str, default="eval_results.json",
                       help="Output JSON file path")
    parser.add_argument("--record", action="store_true",
                       help="Record trajectory for visualization")
    parser.add_argument("--record-output", type=str, default="runs/viz/trajectory.pkl",
                       help="Output path for recorded trajectory")
    parser.add_argument("--grid-size", type=int, default=20,
                       help="Grid size (for baseline)")
    parser.add_argument("--num-agents", type=int, default=4,
                       help="Number of agents (for baseline)")
    parser.add_argument("--fov-size", type=int, default=5,
                       help="FOV size (for baseline)")
    parser.add_argument("--max-steps", type=int, default=128,
                       help="Max steps per episode (for baseline)")
    return parser.parse_args()


def create_env_from_args(args) -> VectorGridWorld:
    """Create VectorGridWorld environment from command-line arguments."""
    config = GridWorldConfig(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        fov_size=args.fov_size,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    # For baselines, use single environment
    num_envs = 1 if args.baseline is not None else args.num_envs
    return VectorGridWorld(config, num_envs=num_envs)


def run_baseline(args, env) -> dict:
    """Run frontier or random baseline evaluation."""
    if args.baseline == 'frontier':
        print(f"Running Frontier Baseline...")
        print(f"  Mode: {'Global (upper bound)' if args.global_knowledge else 'Local (decentralized)'}")
        print(f"  Dispersion weight: {args.dispersion_weight}")
        print(f"  Episodes: {args.num_episodes}")
        print(f"  Max steps: {args.max_steps}")
        
        if args.record:
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
    elif args.baseline == 'random':
        print(f"Running Random Baseline...")
        print(f"  Episodes: {args.num_episodes}")
        print(f"  Max steps: {args.max_steps}")
        return run_random_baseline(env, num_episodes=args.num_episodes, max_steps=args.max_steps, seed=args.seed)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")


def run_frontier_recorded(args, env) -> dict:
    """Run frontier baseline with trajectory recording."""
    from cce.baselines.frontier import FrontierBaseline, FrontierBaselineGlobal
    
    # Setup baseline
    if args.global_knowledge:
        baseline = FrontierBaselineGlobal(
            grid_size=env.G,
            num_agents=env.N,
            fov_radius=env.r,
            dispersion_weight=args.dispersion_weight
        )
    else:
        baseline = FrontierBaseline(
            grid_size=env.G,
            num_agents=env.N,
            fov_radius=env.r,
            dispersion_weight=args.dispersion_weight,
            comm_range=env.cfg.comm_range
        )
    
    # Setup recorder
    grid = env.maps[0].copy()  # (G, G)
    recorder = EpisodeRecorder(env, grid)
    
    # Run single episode
    obs = env.reset()
    recorder.reset()
    
    # Get initial global state for global baseline
    if args.global_knowledge:
        global_state = env.get_global_state()
        global_visited = global_state['visited']
    else:
        global_visited = env.visited_global[0].copy()
    
    for step in range(args.max_steps):
        # Get observations
        if args.global_knowledge:
            global_state = env.get_global_state()
            global_visited = global_state['visited']
        
        # Compute actions
        if args.global_knowledge:
            actions_list = baseline.act(
                agent_positions=[(env.agent_pos[0, i, 0], env.agent_pos[0, i, 1]) for i in range(env.N)],
                visited_masks=[env.visited_local[0, i] for i in range(env.N)],
                global_visited=global_visited
            )
        else:
            actions_list = baseline.act(
                agent_positions=[(env.agent_pos[0, i, 0], env.agent_pos[0, i, 1]) for i in range(env.N)],
                visited_masks=[env.visited_local[0, i] for i in range(env.N)],
                global_visited=global_visited
            )
        
        # Step environment (no send, no messages)
        actions = np.array(actions_list).reshape(1, env.N)
        send_flags = np.zeros((1, env.N), dtype=bool)
        message_embeddings = np.zeros((1, env.N, 8), dtype=np.float32)
        
        obs, rewards, done, info = env.step(actions, send_flags, message_embeddings)
        
        # Update global visited
        if args.global_knowledge:
            global_state = env.get_global_state()
            global_visited = global_state['visited']
        
        coverage = info['coverage'][0]
        
        # Extract messages
        messages = extract_messages_from_info(info, env)
        
        # Record step
        recorder.record_step(obs, rewards, info, messages)
        
        if done[0]:
            break
    
    # Save trajectory
    output_path = Path(args.record_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recorder.save(str(output_path))
    print(f"Trajectory saved to {output_path} ({len(recorder.trajectory)} steps)")
    
    # Return final metrics
    final_coverage = info['coverage'][0]
    from cce.baselines.frontier import compute_overlap
    overlap = compute_overlap([env.visited_local[0, i] for i in range(env.N)])
    
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


def run_mappo(args) -> dict:
    """Run MAPPO evaluation (original evaluate.py logic)."""
    if args.run_dir is None:
        raise ValueError("--run_dir is required for MAPPO evaluation")
    
    run_dir = Path(args.run_dir)
    set_seed(args.seed)

    # Load args from training
    with open(run_dir / "args.json") as f:
        train_args = json.load(f)

    # Reconstruct env config
    env_cfg = GridWorldConfig(
        grid_size=train_args["grid_size"],
        num_agents=train_args["num_agents"],
        fov_size=train_args["fov_size"],
        obstacle_density=train_args["obstacle_density"],
        max_steps=train_args["max_steps"],
        seed=args.seed,
        message_embed_dim=train_args["code_dim"],
    )

    env = VectorGridWorld(env_cfg, num_envs=args.num_envs)

    # Reconstruct agent
    agent = MAPPO(
        env=env,
        actor_hidden=train_args["actor_hidden"],
        critic_hidden=train_args["critic_hidden"],
        backbone=train_args["backbone"],
        gru_hidden=train_args["gru_hidden"],
        n_codes=train_args["n_codes"],
        code_dim=train_args["code_dim"],
        n_slots=train_args["n_slots"],
        target_beta=train_args["beta"],
        budget_mode=train_args["budget_mode"],
        critic_type=train_args["critic_type"],
    )

    # Load checkpoint
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=agent.device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.actor.eval()
    agent.critic.eval()

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"Evaluating over {args.num_episodes} episodes with {args.num_envs} parallel envs...")

    # Collect episodes
    all_coverages = []
    all_t95 = []
    all_rewards = []
    all_comm_costs = []
    all_message_indices = []
    all_send_flags = []

    n_batches = (args.num_episodes + args.num_envs - 1) // args.num_envs

    for batch in range(n_batches):
        obs = env.reset()
        rnn_hidden = agent.actor.init_hidden(args.num_envs * env.N, agent.device).cpu().numpy()
        if agent.backbone == "gru":
            rnn_hidden = rnn_hidden.reshape(args.num_envs, env.N, agent.gru_hidden_dim)
        else:
            rnn_hidden = None

        agent.actor.budget_controller.reset_episode()

        episode_rewards = np.zeros(args.num_envs)
        message_indices_ep = []
        send_flags_ep = []

        for t in range(env.cfg.max_steps):
            with torch.no_grad():
                actions_move, actions_send, log_probs, values, info = agent.select_actions(obs, rnn_hidden)

            # Build message embeddings: (B, N, code_dim)
            if "message_quantised" in info:
                message_embeddings = info["message_quantised"].mean(axis=-2)
            else:
                msg_indices_t = torch.tensor(info["message_indices"], device=agent.device)
                with torch.no_grad():
                    message_embeddings = agent.actor.vq.lookup(msg_indices_t).mean(dim=-2).cpu().numpy()

            obs, rewards, dones, env_infos = env.step(
                actions_move, actions_send.astype(bool), message_embeddings
            )

            episode_rewards += rewards[:, 0]

            # Track messages
            message_indices_ep.append(info["message_indices"])
            send_flags_ep.append(actions_send)

            if rnn_hidden is not None and "rnn_hidden" in info:
                rnn_hidden = info["rnn_hidden"]

            if dones.any():
                for b in range(args.num_envs):
                    if dones[b]:
                        all_coverages.append(float(env_infos["coverage"][b]))
                        all_t95.append(int(env_infos["t95"][b]))
                        all_rewards.append(float(episode_rewards[b]))
                        episode_rewards[b] = 0

        # Concatenate message history
        if message_indices_ep:
            all_message_indices.append(np.stack(message_indices_ep, axis=0))
            all_send_flags.append(np.stack(send_flags_ep, axis=0))

    # Compute metrics
    all_coverages = np.array(all_coverages[:args.num_episodes])
    all_t95 = np.array(all_t95[:args.num_episodes])
    all_rewards = np.array(all_rewards[:args.num_episodes])
    all_comm_costs = np.zeros(len(all_coverages))  # approximate

    metrics = compute_episode_metrics(
        coverages=all_coverages,
        t95_values=all_t95,
        comm_costs=all_comm_costs,
        rewards=all_rewards,
        max_steps=env.cfg.max_steps,
    )

    # Message statistics
    if all_message_indices:
        msg_indices = np.concatenate(all_message_indices, axis=1)
        send_flags = np.concatenate(all_send_flags, axis=1)
        msg_stats = message_statistics(msg_indices, send_flags, train_args["n_codes"])
        metrics.update(msg_stats)

    return metrics


def print_metrics(metrics: dict, label: str = "Metrics"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"{'Metric':<25} {'Value':>15}")
    print(f"{'-'*25} {'-'*15}")
    
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


def main() -> None:
    args = parse_args()
    
    if args.baseline is not None:
        # Run baseline
        env = create_env_from_args(args)
        print(f"Environment: {args.grid_size}x{args.grid_size} grid, {args.num_agents} agents, "
              f"FOV {args.fov_size}, {args.max_steps} steps")
        
        metrics = run_baseline(args, env)
        label = f"{args.baseline.capitalize()} Baseline"
        if args.baseline == 'frontier' and args.global_knowledge:
            label += " (Global)"
        
        print_metrics(metrics, label)
        
        if args.output:
            save_metrics(metrics, Path(args.output))
    else:
        # Run MAPPO
        metrics = run_mappo(args)
        print_metrics(metrics, "MAPPO Evaluation")
        
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