# Quick Start Guide

## Installation (2 minutes)

```bash
cd /Users/srimaan/Sem7/FAI/Project
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Tests (1 minute)

```bash
pytest tests/ -v
```

## Smoke Test (5 minutes on CPU)

```bash
bash scripts/run_smoke.sh
```

This runs a tiny training job to verify everything works. Check `runs/smoke/eval_results.json` for metrics.

## Full Training (2-4 hours on Kaggle T4)

```bash
bash scripts/run_budget_sweep.sh
```

This trains across β ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0} × 5 seeds. Results in `runs/budget_sweep/` and analysis in `analysis/`.

## Evaluate a Trained Model

```bash
python evaluate.py \
    --run_dir runs/smoke \
    --num_episodes 50 \
    --num_envs 16 \
    --seed 42
```

## Analyse Results

```bash
python analyse.py \
    --results_dir runs/budget_sweep \
    --output_dir analysis
```

Generates Pareto frontier plots and summary tables.

---

## Project Structure

```
cce/                          # Main package
├── envs/                     # Vectorised grid-world environment
│   ├── mapgen.py            # Procedural map generation
│   └── grid_world.py        # Core environment (B parallel envs)
├── comm/                     # Communication modules
│   ├── encoder.py           # Observation encoder (MLP/GRU)
│   ├── vq.py                # Vector quantizer with EMA updates
│   ├── channel.py           # Message encoder/decoder
│   └── budget.py            # Lagrangian budget controller
├── algo/                     # MAPPO algorithm
│   ├── distributions.py     # Action distributions
│   ├── networks.py          # Actor + critic networks
│   ├── buffer.py            # Rollout buffer with GAE
│   └── mappo.py             # Main MAPPO trainer
├── eval/                     # Evaluation tools
│   ├── metrics.py           # Coverage, T95, overlap, etc.
│   ├── pareto.py            # Pareto frontier + Kneedle
│   └── protocol.py          # Message entropy, MI analysis
└── utils/                    # Utilities
    ├── config.py            # Hydra/OmegaConf helpers
    ├── logging.py           # W&B + local logging
    └── seeding.py           # Reproducibility

train.py                      # Main training script
evaluate.py                   # Post-training evaluation
analyse.py                    # Multi-run analysis
configs/                      # Hydra YAML configs
tests/                        # Pytest tests
scripts/                      # Shell scripts for experiments
```

---

## Key Design Decisions

### Environment
- **Vectorised over B parallel envs** (pure NumPy, no IPC overhead)
- **Fixed-length episodes** (simplifies PPO buffer)
- **Local visited masks** (no global info leakage to agents)
- **Manhattan-distance communication** (agents within range R=3 can communicate)

### MAPPO
- **Shared actor** across homogeneous agents
- **Centralised critic** with CNN over global state (3-channel image + positions)
- **GAE** with λ=0.95, value normalisation (PopArt-lite)
- **Supports MLP or GRU** backbones (GRU for partial observability)

### Communication
- **VQ bottleneck** with EMA codebook updates and dead-code revival
- **Messages** = d_m discrete codes from codebook of size K (default: 4 codes × 16 codebook = 16 bits)
- **Budget control** via Lagrangian dual with PID controller
- **Two modes**: policy (send decision sampled) or utility (deterministic threshold)

### Budget Mechanism
- **Normalised cost**: C_comm_t = (n_send_t * d_m) / (N * d_m^max)
- **Target**: mean(C_comm_t) ≤ β over episode
- **Enforcement**: PID controller adjusts λ (policy mode) or τ (utility mode)
- **Hard cutoff**: silence when budget exhausted

---

## Hyperparameters (from paper Table 2)

| Parameter | Value | Location |
|-----------|-------|----------|
| Actor LR | 3e-4 | `--lr_actor` |
| Critic LR | 1e-3 | `--lr_critic` |
| γ | 0.99 | `--gamma` |
| GAE λ | 0.95 | `--gae_lambda` |
| PPO clip ε | 0.2 | `--clip_eps` |
| Entropy coef | 0.01 | `--entropy_coef` |
| Hidden dim (MLP) | 64 | `--actor_hidden` |
| Hidden dim (GRU) | 128 | `--gru_hidden` |
| Codebook size K | 16 | `--n_codes` |
| Code dim | 8 | `--code_dim` |
| Message slots d_m | 4 | `--n_slots` |

---

## Running on Kaggle (Recommended for Long Runs)

1. Zip the project: `cd /Users/srimaan/Sem7/FAI/Project && zip -r cce_project.zip .`
2. Upload to Kaggle as a dataset
3. Create a notebook, attach the dataset, install dependencies:
   ```python
   !pip install -e /kaggle/input/cce_project/cce_project
   ```
4. Run training:
   ```python
   !cd /kaggle/input/cce_project/cce_project && bash scripts/run_budget_sweep.sh
   ```
5. Download results from `/kaggle/working/`

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'cce'"**
→ Run `pip install -e .` from the project root.

**"CUDA out of memory"**
→ Reduce `--num_envs` (e.g., 32 instead of 64) or `--batch_size` (e.g., 1600 instead of 3200).

**"Coverage stays at 0"**
→ Check that `obs_neighbours=True` and `obs_message_embed=True` in `GridWorldConfig`. Also verify the reward weights (`w_coverage` should be positive).

**"Budget not respected (actual β ≠ target β)"**
→ Increase `--num_updates` (budget controller needs time to converge). Or tune PID gains: `pid_kp=1.0, pid_ki=0.1`.

**"VQ perplexity stays low (codebook collapse)"**
→ Reduce `--n_codes` (e.g., 8 instead of 16) or increase `--entropy_coef` (e.g., 0.05).

---

## Next Steps for Srimaan

1. **Run the smoke test** to verify installation
2. **Read the code** (start with `cce/envs/grid_world.py`, then `cce/algo/mappo.py`)
3. **Run a short training** (100 updates) on your local machine
4. **Inspect W&B logs** to see learning curves
5. **Tune hyperparameters** if needed (start with `--entropy_coef` and `--lr_actor`)
6. **Run the full β-sweep** on Kaggle (2-4 hours)
7. **Analyse results** with `analyse.py` and generate plots for the paper

---

## Contact

Questions about the code? Ask ChatGPT (this project was built with extensive AI assistance). Key modules to reference:
- **VQ bottleneck**: `cce/comm/vq.py` (lines 1-180)
- **Budget controller**: `cce/comm/budget.py` (lines 1-100)
- **MAPPO update**: `cce/algo/mappo.py` (lines 200-350)
- **Environment step**: `cce/envs/grid_world.py` (lines 150-280)
