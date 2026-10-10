"""Main training script for MAPPO with communication constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cce.envs.grid_world import VectorGridWorld, GridWorldConfig
from cce.algo.mappo import MAPPO
from cce.utils.seeding import set_seed
from cce.utils.logging import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAPPO with communication constraints")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_updates", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--run_dir", type=str, default="runs/exp1")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.25, help="Target communication budget β")

    # Environment
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--fov_size", type=int, default=5)
    parser.add_argument("--obstacle_density", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=128)

    # Model
    parser.add_argument("--backbone", type=str, default="mlp", choices=["mlp", "gru"])
    parser.add_argument("--actor_hidden", type=int, default=64)
    parser.add_argument("--critic_hidden", type=int, default=256)
    parser.add_argument("--gru_hidden", type=int, default=128)
    parser.add_argument("--n_codes", type=int, default=16)
    parser.add_argument("--code_dim", type=int, default=8)
    parser.add_argument("--n_slots", type=int, default=4, help="d_m: number of code slots per message")
    parser.add_argument("--budget_mode", type=str, default="policy", choices=["policy", "utility"])
    parser.add_argument("--critic_type", type=str, default="cnn", choices=["mlp", "cnn"])

    # Training
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=3200)
    parser.add_argument("--num_epochs", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Create run directory
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Environment config
    env_cfg = GridWorldConfig(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        fov_size=args.fov_size,
        obstacle_density=args.obstacle_density,
        max_steps=args.max_steps,
        seed=args.seed,
        message_embed_dim=args.code_dim,
    )

    env = VectorGridWorld(env_cfg, num_envs=args.num_envs)

    # Logger
    logger = Logger(
        run_dir=run_dir,
        use_wandb=args.use_wandb,
        run_name=args.run_name,
        config=vars(args),
    )

    # Algorithm
    agent = MAPPO(
        env=env,
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        backbone=args.backbone,
        gru_hidden=args.gru_hidden,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        n_codes=args.n_codes,
        code_dim=args.code_dim,
        n_slots=args.n_slots,
        target_beta=args.beta,
        budget_mode=args.budget_mode,
        critic_type=args.critic_type,
    )

    print(f"Training with β={args.beta}, {args.num_envs} envs, seed={args.seed}")
    print(f"Device: {agent.device}")
    print(f"Parameters: actor={sum(p.numel() for p in agent.actor.parameters()):,}, "
          f"critic={sum(p.numel() for p in agent.critic.parameters()):,}")

    best_coverage = 0.0

    for update in range(1, args.num_updates + 1):
        stats = agent.train_step(logger if update % args.log_interval == 0 else None)

        if update % args.log_interval == 0:
            print(
                f"[{update}/{args.num_updates}] "
                f"cov={stats['coverage']:.3f} t95={stats['t95']:.1f} "
                f"β={stats['actual_beta']:.3f} λ={stats['lambda']:.3f} "
                f"actor_loss={stats['actor_loss']:.4f} "
                f"entropy={stats['entropy']:.3f}"
            )

        if stats["coverage"] > best_coverage:
            best_coverage = stats["coverage"]
            logger.save_model(
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "update": update,
                    "stats": stats,
                },
                "best_model.pt",
            )

        if update % args.save_interval == 0:
            logger.save_model(
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "update": update,
                },
                f"model_{update}.pt",
            )

    logger.log_summary({"best_coverage": best_coverage})
    logger.finish()
    print(f"Training complete. Best coverage: {best_coverage:.3f}")


if __name__ == "__main__":
    main()
