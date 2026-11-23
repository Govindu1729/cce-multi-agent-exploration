# Quick Start Guide

## Installation

### Linux / macOS (2 minutes)
```bash
# Clone and enter directory
cd cce-multi-agent-exploration
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Windows (PowerShell)
```powershell
cd cce-multi-agent-exploration
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

---

## 1. Interactive Web Dashboard (Streamlit)

Launch the trajectory replay viewer and multi-run comparison dashboard:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 2. Run Tests (1 minute)

```bash
pytest tests/ -v
```
Or run the dashboard automated self-test:
```bash
python test_app.py
```

---

## 3. Smoke Test (5 minutes on CPU)

```bash
bash scripts/run_smoke.sh
```
Or directly via python:
```bash
python train.py --num_envs 16 --num_updates 50 --grid_size 10 --num_agents 4 --fov_size 3 --max_steps 32 --run_dir runs/smoke
```
This runs a tiny training job to verify everything works. Check `runs/smoke/eval_results.json` for metrics.

---

## 4. Full Training (2-4 hours on Kaggle T4)

```bash
bash scripts/run_budget_sweep.sh
```
This trains across $\beta \in \{0, 0.1, 0.25, 0.5, 0.75, 1.0\} \times 5$ seeds. Results in `runs/budget_sweep/` and analysis in `analysis/`.

---

## 5. Evaluate Baselines & Trained Models

```bash
# Evaluate Frontier baseline and record trajectory for Streamlit
python scripts/evaluate.py --baseline frontier --record --record_output runs/viz/frontier_trajectory.pkl

# Evaluate Random baseline
python scripts/evaluate.py --baseline random --record --record_output runs/viz/random_trajectory.pkl

# Evaluate a trained MAPPO checkpoint
python scripts/evaluate.py --baseline mappo --run_dir runs/smoke --checkpoint checkpoints/best_model.pt --num_episodes 10
```

---

## 6. Analyse Multi-Run Results

```bash
python analyse.py --results_dir runs/budget_sweep --output_dir analysis
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

app.py                        # Streamlit web dashboard
train.py                      # Main MAPPO training script
evaluate.py                   # Post-training evaluation
analyse.py                    # Multi-run analysis
configs/                      # Hydra YAML configs
tests/                        # Pytest tests (27 unit tests)
scripts/                      # Shell scripts for experiments
```

---

## Key Design Decisions

### Environment
- **Vectorised over B parallel envs** (pure NumPy, no IPC overhead)
- **Fixed-length episodes** (simplifies PPO buffer)
- **Local visited masks** (no global info leakage to agents)
- **Manhattan-distance communication** (agents within range $R=3$ can communicate)

### MAPPO
- **Shared actor** across homogeneous agents
- **Centralised critic** with CNN over global state (3-channel image + positions)
- **GAE** with $\lambda=0.95$, value normalisation (PopArt-lite)
- **Supports MLP or GRU** backbones (GRU for partial observability)

### Communication
- **VQ bottleneck** with EMA codebook updates and dead-code revival
- **Messages** = $d_m$ discrete codes from codebook of size $K$ (default: 4 codes $\times$ 16 codebook = 16 bits)
- **Budget control** via Lagrangian dual with PID controller
- **Two modes**: policy (send decision sampled) or utility (deterministic threshold)

### Budget Mechanism
- **Normalised cost**: $C_{\text{comm}, t} = \frac{n_{\text{send}, t} \cdot d_m}{N \cdot d_m^{\max}}$
- **Target**: $\text{mean}(C_{\text{comm}, t}) \le \beta$ over episode
- **Enforcement**: PID controller adjusts $\lambda$ (policy mode) or $\tau$ (utility mode)
- **Hard cutoff**: silence when budget exhausted

---

## Running on Kaggle (Recommended for Long Runs)

1. Zip the project repository: `zip -r cce_project.zip . -x "*.pyc" "__pycache__/*" ".venv/*"`
2. Upload to Kaggle as a dataset.
3. Create a notebook, attach the dataset, and install dependencies:
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
