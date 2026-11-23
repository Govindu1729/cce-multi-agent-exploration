# Communication-Constrained Multi-Agent Exploration (CCE)

Cooperative multi-agent exploration on grid worlds under a normalised communication budget $\beta$.

## Authors (Team 37)

- **Govindu Srimaan** (23110121) — MAPPO & Communication Bottleneck
- **Maru Jainil** (23110200) — Vectorized GridWorld Environment & Rewards
- **Manavi** (24110193) — Training Pipeline & $\beta$-Sweep Experiments
- **Priyanka** (24110278) — Evaluation Metrics, Pareto/Knee Analysis & Protocol Evaluation

## Research Question

> **How does a constrained communication budget $\beta$ affect the efficiency of cooperative multi-agent exploration?**

## Architecture

- **Environment:** Vectorised grid-world ($B$ parallel environments in pure NumPy) with procedurally generated maps, connectivity guarantees, and partial observability via $k \times k$ FOV.
- **Algorithm:** MAPPO with shared actor parameters and a centralised critic ($V_\phi(s_t)$ over global state). Supports both MLP and GRU backbones.
- **Communication Bottleneck:** Differentiable discrete communication bottleneck with EMA-updated Vector Quantization (VQ) codebook and straight-through gradient estimator. Messages consist of $d_m$ discrete codes from a codebook of size $K$.
- **Budget Control:** Lagrangian dual mechanism enforcing budget $\beta$ via dynamic multipliers and utility gating ($u_i^t > \tau$).
- **Evaluation & Visualisation:** Full evaluation suite tracking Coverage $Cov(t)$, $T_{95}$, Overlap, Bandwidth, and Communication Efficiency. Empirical Pareto frontier analysis with Kneedle knee detection. Interactive Streamlit dashboard (`app.py`) for frame-by-frame trajectory replay.

---

## Installation

### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install streamlit plotly pytest
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install streamlit plotly pytest
```

---

## Quick Start & Usage

### 1. Interactive Web Dashboard (`app.py`)
Launch the trajectory replay viewer and multi-run Pareto comparison dashboard:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 2. Verify with Unit Tests
Run the 27 unit tests across environment, VQ bottleneck, budget controller, and MAPPO:
```bash
pytest tests/ -v
```
Or run the dashboard automated self-test:
```bash
python test_app.py
```

### 3. Evaluate Baselines & Record Trajectories
```bash
# Frontier exploration baseline (records trajectory for Streamlit)
python scripts/evaluate.py --baseline frontier --num_episodes 1 --record --record_output runs/viz/frontier_trajectory.pkl

# Random exploration baseline
python scripts/evaluate.py --baseline random --num_episodes 1 --record --record_output runs/viz/random_trajectory.pkl
```

### 4. Training MAPPO Models
```bash
# Quick smoke test (2-5 min on CPU)
python train.py --run_dir runs/smoke --num_envs 16 --num_updates 50 --max_steps 32

# Train with communication budget β = 0.25
python train.py --beta 0.25 --run_dir runs/budget_sweep/beta_0.25 --num_updates 500
```

### 5. Multi-Run Pareto Frontier Analysis
```bash
python analyse.py --results_dir runs/budget_sweep --output_dir analysis
```
Generates `analysis/pareto_frontier.png`, `analysis/entropy_vs_beta.png`, and `analysis/results_table.csv`.

---

## Project Structure

```
cce/                          # Main package (communication-constrained exploration)
├── envs/                     # Vectorised grid-world (maps, FOV, connectivity)
├── comm/                     # Encoder, VQ codebook, channel, budget controller
├── algo/                     # MAPPO trainer, buffer, networks, distributions
├── eval/                     # Metrics, Pareto frontier + Kneedle, protocol analysis
└── utils/                    # Configs, logging (W&B / local), seeding
app.py                        # Streamlit interactive web dashboard
train.py                      # Main MAPPO training script
evaluate.py                   # Post-training evaluation script
analyse.py                    # Multi-run Pareto and entropy analysis
configs/                      # Experiment YAML configs (Hydra)
tests/                        # Pytest unit tests (27 tests)
scripts/                      # Shell execution scripts
```
