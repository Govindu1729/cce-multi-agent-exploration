#!/usr/bin/env python3
"""Unified evaluation entry point: trained MAPPO, frontier, or random policy.

Produces engine-comparable metrics — Coverage, T95, Overlap, Bandwidth — and
optionally records a trajectory pickle for the Streamlit dashboard.

The learning code (``cce/algo/``) and communication code (``cce/comm/``) are
used strictly read-only; no file in either package is modified.

Examples
--------
Evaluate the β sweep checkpoint::

    python scripts/evaluate.py --baseline mappo \\
        --run_dir runs/beta_sweep \\
        --checkpoint checkpoints/best_model.pt \\
        --num_episodes 10 --seed 42

Run the frontier baseline (decentralised)::

    python scripts/evaluate.py --baseline frontier --num_episodes 10

Run the frontier baseline and record a dashboard trajectory::

    python scripts/evaluate.py --baseline frontier --num_episodes 1 \\
        --record --record_output runs/viz/trajectory.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make the project importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cce.envs.grid_world import GridWorldConfig, VectorGridWorld
from cce.eval.metrics import compute_episode_metrics, compute_overlap
from cce.eval.recorder import EpisodeRecorder
from cce.baselines.frontier import make_baseline
from cce.utils.seeding import set_seed

try:  # optional — only needed for MAPPO evaluation
    import torch
    from cce.algo.mappo import MAPPO
except Exception:  # pragma: no cover
    torch = None
    MAPPO = None


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MAPPO or a classical baseline")
    p.add_argument("--baseline", type=str, default="mappo",
                   choices=["mappo", "frontier", "random"],
                   help="Which policy to evaluate (default: mappo)")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Training run directory (required for --baseline mappo)")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                   help="Checkpoint path, relative to --run_dir or absolute")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="eval_results.json")
    p.add_argument("--dispersion_weight", type=float, default=1.0,
                   help="Frontier baseline dispersion heuristic weight")
    p.add_argument("--dm_bits", type=int, default=4,
                   help="Message dimension in bits, for the bandwidth metric")
    p.add_argument("--record", action="store_true",
                   help="Record one episode to a pickle for the dashboard")
    p.add_argument("--record_output", type=str, default="runs/viz/trajectory.pkl")

    # Environment overrides (used for baselines, and MAPPO when no args.json)
    p.add_argument("--grid_size", type=int, default=20)
    p.add_argument("--num_agents", type=int, default=4)
    p.add_argument("--fov_size", type=int, default=5)
    p.add_argument("--obstacle_density", type=float, default=0.2)
    p.add_argument("--max_steps", type=int, default=128)
    return p.parse_args()


# ----------------------------------------------------------------------
# Environment construction
# ----------------------------------------------------------------------
def build_env_from_args(args) -> VectorGridWorld:
    cfg = GridWorldConfig(
        grid_size=args.grid_size,
        num_agents=args.num_agents,
        fov_size=args.fov_size,
        obstacle_density=args.obstacle_density,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    # Baselines step a single environment; MAPPO can batch.
    num_envs = 1 if args.baseline in ("frontier", "random") else args.num_envs
    return VectorGridWorld(cfg, num_envs=num_envs)


def build_mappo(args):
    """Reconstruct env + agent from a training run's args.json and load ckpt."""
    if MAPPO is None or torch is None:
        raise RuntimeError("PyTorch / MAPPO unavailable; cannot evaluate a checkpoint")

    run_dir = Path(args.run_dir)
    with open(run_dir / "args.json") as fh:
        train_args = json.load(fh)

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

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path
        if not ckpt_path.exists():
            ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=agent.device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.actor.eval()
    agent.critic.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    return env, agent, train_args


# ----------------------------------------------------------------------
# Episode collection
# ----------------------------------------------------------------------
def _episode_overlap(env: VectorGridWorld, b: int) -> float:
    vg = env.visited_global[b:b + 1]                 # (1, G, G)
    vl = env.visited_local[b:b + 1]                  # (1, N, G, G)
    return float(compute_overlap(vg, vl))


def collect_episodes(env, args, mappo_agent=None, train_args=None) -> dict:
    """Run ``num_episodes`` and return aggregate metrics."""
    B = env.B
    N = env.N
    d_code = env.cfg.message_embed_dim
    max_steps = env.cfg.max_steps

    baseline = None
    if args.baseline in ("frontier", "random"):
        baseline = make_baseline(args.baseline, env, seed=args.seed,
                                 dispersion_weight=args.dispersion_weight)

    recorder = None
    if args.record:
        recorder = EpisodeRecorder(env, env.maps[0], batch_index=0, dm_bits=args.dm_bits)

    coverages: list[float] = []
    t95_values: list[int] = []
    rewards_ep: list[float] = []
    comm_costs_ep: list[float] = []
    overlaps: list[float] = []
    bandwidth_bits_ep: list[float] = []

    n_batches = (args.num_episodes + B - 1) // B

    for batch in range(n_batches):
        obs = env.reset()
        if recorder is not None and batch == 0:
            recorder.reset()

        ep_reward = np.zeros(B, dtype=np.float64)
        ep_comm_cost = np.zeros(B, dtype=np.float64)
        ep_bits = np.zeros(B, dtype=np.float64)
        finished = np.zeros(B, dtype=bool)

        # MAPPO hidden state / budget reset
        rnn_hidden = None
        if mappo_agent is not None:
            rnn_hidden = mappo_agent.actor.init_hidden(B * N, mappo_agent.device).cpu().numpy()
            if mappo_agent.backbone == "gru":
                rnn_hidden = rnn_hidden.reshape(B, N, mappo_agent.gru_hidden_dim)
            else:
                rnn_hidden = None
            if hasattr(mappo_agent.actor, "budget_controller"):
                mappo_agent.actor.budget_controller.reset_episode()

        for t in range(max_steps):
            message_indices = None

            if mappo_agent is not None:
                with torch.no_grad():
                    actions_move, actions_send, _, _, info = mappo_agent.select_actions(obs, rnn_hidden)
                if "message_quantised" in info:
                    message_embeddings = info["message_quantised"].mean(axis=-2)
                else:
                    mi = torch.tensor(info["message_indices"], device=mappo_agent.device)
                    with torch.no_grad():
                        message_embeddings = mappo_agent.actor.vq.lookup(mi).mean(dim=-2).cpu().numpy()
                send_flags = actions_send.astype(bool)
                message_indices = info.get("message_indices")
            else:
                actions_move = baseline.act(batch_index=0) if B == 1 else \
                    np.concatenate([baseline.act(batch_index=b) for b in range(B)], axis=0)
                if actions_move.shape != (B, N):
                    actions_move = np.broadcast_to(actions_move, (B, N)).copy()
                send_flags = np.zeros((B, N), dtype=bool)
                message_embeddings = np.zeros((B, N, d_code), dtype=np.float32)

            obs, rewards, dones, infos = env.step(actions_move, send_flags, message_embeddings)

            ep_reward += rewards[:, 0]
            ep_comm_cost += infos["comm_cost"]
            ep_bits += send_flags.sum(axis=1) * args.dm_bits

            if mappo_agent is not None and rnn_hidden is not None and "rnn_hidden" in info:
                rnn_hidden = info["rnn_hidden"]

            # Record the first environment of the first batch.
            if recorder is not None and batch == 0:
                recorder.record_step(obs, infos, send_flags=send_flags,
                                     message_indices=message_indices)

            # Collect completed episodes.
            for b in range(B):
                if dones[b] and not finished[b]:
                    finished[b] = True
                    if len(coverages) < args.num_episodes:
                        coverages.append(float(infos["coverage"][b]))
                        t95_values.append(int(infos["t95"][b]))
                        rewards_ep.append(float(ep_reward[b]))
                        comm_costs_ep.append(float(ep_comm_cost[b] / max_steps))
                        overlaps.append(_episode_overlap(env, b))
                        bandwidth_bits_ep.append(float(ep_bits[b]))
            if finished.all() and len(coverages) >= args.num_episodes:
                break

    # Save recorded trajectory.
    if recorder is not None:
        recorder.save(args.record_output)
        print(f"Trajectory recorded: {args.record_output} ({len(recorder.trajectory)} frames)")

    return _aggregate(coverages, t95_values, rewards_ep, comm_costs_ep,
                      overlaps, bandwidth_bits_ep, max_steps)


def _aggregate(coverages, t95_values, rewards_ep, comm_costs_ep,
               overlaps, bandwidth_bits_ep, max_steps) -> dict:
    coverages = np.asarray(coverages, dtype=np.float64)
    t95_values = np.asarray(t95_values, dtype=np.int64)
    rewards_ep = np.asarray(rewards_ep, dtype=np.float64)
    comm_costs_ep = np.asarray(comm_costs_ep, dtype=np.float64)

    core = compute_episode_metrics(
        coverages=coverages,
        t95_values=t95_values,
        comm_costs=comm_costs_ep,
        rewards=rewards_ep,
        max_steps=max_steps,
    )

    overlaps = np.asarray(overlaps, dtype=np.float64)
    bits = np.asarray(bandwidth_bits_ep, dtype=np.float64)

    core.update({
        "overlap": float(overlaps.mean()) if overlaps.size else 0.0,
        "overlap_std": float(overlaps.std()) if overlaps.size else 0.0,
        "bandwidth_bits": float(bits.mean()) if bits.size else 0.0,
        "bandwidth_bits_std": float(bits.std()) if bits.size else 0.0,
        "num_episodes": int(coverages.size),
    })
    return core


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def print_metrics(metrics: dict, label: str) -> None:
    print(f"\n{'=' * 52}")
    print(f"{label}")
    print(f"{'=' * 52}")
    order = [
        ("coverage", "Coverage (mean)"),
        ("coverage_std", "Coverage (std)"),
        ("t95", "T95 (mean steps)"),
        ("t95_std", "T95 (std)"),
        ("success_rate", "Success rate (>=95%)"),
        ("overlap", "Overlap (mean)"),
        ("overlap_std", "Overlap (std)"),
        ("bandwidth_bits", "Bandwidth (bits/ep)"),
        ("avg_comm_cost", "Avg comm cost (norm)"),
        ("comm_efficiency", "Comm efficiency"),
        ("avg_reward", "Avg reward"),
        ("num_episodes", "Episodes"),
    ]
    for key, name in order:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {name:<24} {val:>12.4f}")
            else:
                print(f"  {name:<24} {val:>12}")
    print(f"{'=' * 52}")


def to_jsonable(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, float) and not np.isfinite(v):
            out[k] = None
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.baseline == "mappo":
        if args.run_dir is None:
            print("ERROR: --run_dir is required for --baseline mappo", file=sys.stderr)
            return 2
        env, agent, train_args = build_mappo(args)
        label = f"MAPPO ({Path(args.run_dir).name})"
        metrics = collect_episodes(env, args, mappo_agent=agent, train_args=train_args)
    else:
        env = build_env_from_args(args)
        label = f"{args.baseline.capitalize()} baseline"
        metrics = collect_episodes(env, args)

    print_metrics(metrics, label)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(to_jsonable(metrics), indent=2))
        print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())