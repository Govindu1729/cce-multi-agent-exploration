"""Evaluate a trained model over multiple episodes and compute final metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cce.envs.grid_world import VectorGridWorld, GridWorldConfig
from cce.algo.mappo import MAPPO
from cce.eval.metrics import compute_episode_metrics
from cce.eval.protocol import message_statistics
from cce.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.run_dir)

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
            # message_quantised has shape (B, N, n_slots, code_dim); mean over slots
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

    # Print results
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    with open(run_dir / args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {run_dir / args.output}")


if __name__ == "__main__":
    main()
