# Communication-Constrained Multi-Agent Exploration

**PRIME AI · Team 37 · 7th Semester**

Cooperative multi-agent exploration on grid worlds under a normalised communication budget β.

## Authors

- **Govindu Srimaan** (23110121) — full implementation (MAPPO, differentiable communication bottleneck with VQ, Lagrangian budget gate, vectorised environment, evaluation pipeline)
- Maru Jainil (23110200)
- Manavi (24110193)
- Priyanka (24110278)

## Research question

> How does a constrained communication budget β affect the efficiency of cooperative multi-agent exploration?

## Architecture

- **Environment:** vectorised grid-world (B parallel envs, pure NumPy) with procedurally generated maps, connectivity guarantees, and partial observability via k×k FOV.
- **Algorithm:** MAPPO with shared actor + centralised critic (CNN over global state). Supports both MLP and GRU actors.
- **Communication:** differentiable discrete bottleneck with EMA-updated VQ codebook and straight-through gradient estimator. Messages are `d_m` discrete codes from a codebook of size `K`.
- **Budget:** Lagrangian-dual mechanism that enforces β exactly via a per-episode token budget, with learnable per-timestep send decisions.
- **Metrics:** coverage, T95, overlap, success rate, actual bandwidth, communication efficiency. Pareto frontier analysed with Kneedle.

## Installation

```bash
cd /Users/srimaan/Sem7/FAI/Project
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

```bash
# Smoke test — 5 min on CPU
bash scripts/run_smoke.sh

# Full β-sweep (main experiment, ~2 h on Kaggle T4)
bash scripts/run_budget_sweep.sh
```

## Configuration

All hyperparameters are in `configs/base.yaml` with overrides in `configs/exp/`. Uses **Hydra**.

## Running tests

```bash
pytest tests/ -v
```

## Project layout

```
cce/              # main package (communication-constrained exploration)
├── envs/         # vectorised grid-world
├── comm/         # encoder, VQ, channel, budget gate
├── algo/         # MAPPO, buffer, networks, distributions
├── eval/         # metrics, Pareto, protocol analysis
└── utils/        # config, seeding, logging
train.py          # main training script
evaluate.py       # post-training evaluation
analyse.py        # Pareto + message protocol analysis
configs/          # Hydra YAML configs
tests/            # pytest tests for every module
```
