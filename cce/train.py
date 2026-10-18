"""
Training entry point for FAI multi-agent coverage experiments.

This module provides the main training loop that can be invoked via Hydra configs.
"""

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, '/Users/srimaan/.openclaw/workspaces/repo-cce')

from cce.envs.grid_world import GridWorld, GridWorldConfig
from cce.baselines.frontier import run_frontier_baseline
from cce.eval.metrics import compute_metrics, aggregate_episode_result


def create_env_from_config(cfg: DictConfig) -> GridWorld:
    """Create GridWorld environment from Hydra config."""
    env_config = GridWorldConfig(
        grid_size=cfg.env.grid_size,
        num_agents=cfg.env.num_agents,
        fov_radius=cfg.env.fov_radius,
        max_steps=cfg.env.max_steps,
        seed=cfg.env.seed,
        # Decoupled communication constraints
        max_dm=cfg.comm.max_dm,
        comm_frequency=cfg.comm.comm_frequency,
        topology=cfg.comm.topology,
        comm_budget=cfg.comm.max_tokens,
        comm_range=cfg.comm.comm_range,
        tokens_per_send=cfg.comm.tokens_per_send,
    )
    return GridWorld(env_config)


@hydra.main(version_base=None, config_path="../../configs/experiment", config_name="base")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    print("=" * 60)
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Description: {cfg.experiment.description}")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seeds
    seed = cfg.experiment.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = create_env_from_config(cfg)
    
    # Run baseline evaluation (since we don't have MAPPO implemented yet)
    print("\nRunning frontier baseline evaluation...")
    metrics = run_frontier_baseline(
        env,
        num_episodes=cfg.experiment.num_eval_episodes,
        max_steps=cfg.experiment.max_steps,
        seed=seed
    )
    
    print("\nBaseline Metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
    
    # Save metrics
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    import json
    with open(os.path.join(output_dir, "baseline_metrics.json"), "w") as f:
        # Convert numpy types to Python types
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (np.float32, np.float64)):
                serializable[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                serializable[k] = int(v)
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2)
    
    print(f"\nMetrics saved to {output_dir}/baseline_metrics.json")
    print("\nTraining complete!")


def train_with_config(config_path: str, overrides: list = None):
    """Train with a specific config file, handling defaults manually."""
    if overrides is None:
        overrides = []
    
    # Load base config
    base_cfg = OmegaConf.load("configs/experiment/base.yaml")
    
    # Load sweep config
    sweep_cfg = OmegaConf.load(config_path)
    
    # Merge configs (sweep overrides base)
    cfg = OmegaConf.merge(base_cfg, sweep_cfg)
    
    # Apply command-line overrides
    for override in overrides:
        key, value = override.split("=")
        OmegaConf.update(cfg, key, value)
    
    print("=" * 60)
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Description: {cfg.experiment.description}")
    print("=" * 60)
    
    # Set seeds
    seed = cfg.experiment.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = create_env_from_config(cfg)
    
    # Run baseline evaluation
    print("\nRunning frontier baseline evaluation...")
    metrics = run_frontier_baseline(
        env,
        num_episodes=cfg.experiment.num_eval_episodes,
        max_steps=cfg.experiment.max_steps,
        seed=seed
    )
    
    print("\nBaseline Metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
    
    return metrics


if __name__ == "__main__":
    # Check if running with hydra or directly
    if len(sys.argv) > 1 and sys.argv[1] == "--config":
        config_path = sys.argv[2]
        overrides = sys.argv[3:] if len(sys.argv) > 3 else []
        train_with_config(config_path, overrides)
    else:
        train()