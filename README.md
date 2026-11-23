
# Communication-Constrained Multi-Agent Exploration (CCE)

Cooperative multi-agent exploration in unknown grid worlds under a normalised communication budget $\beta \in [0, 1]$.

## Authors & Contributions (Team 37)

| Member | Roll No. | Assigned Module | Major Tasks Assigned | Task Fulfilled |
| :--- | :---: | :--- | :--- | :---: |
| **Govindu Srimaan** | 23110121 | MAPPO & Comm. Bottleneck | Shared actor/centralised critic, rollout buffer alignment fix, VQ-STE bottleneck, two-layer `TokenBucket` + KKT $\lambda$ budget controller | Yes (100%) |
| **Maru Jainil** | 23110200 | Environment & Baselines | Vectorised $20\times 20$ `VectorGridWorld`, flood-fill map connectivity, private `visited_local` masks, reward shaping, Frontier baseline fix | Yes (100%) |
| **Manavi** | 24110193 | Training & $\beta$-Sweep | Hydra configuration, 300-update training sweep across $\beta \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$, checkpoint calibration, 37-test verification | Yes (100%) |
| **Priyanka Ralebhat** | 24110278 | Evaluation & Analysis | Standardised 20-episode evaluation suite, $T_{95}$ & overlap metrics, Pareto/Kneedle knee detection, spatial overlap heatmaps, final plots | Yes (100%) |

---

## Research Question
> **How does a constrained communication budget $\beta$ affect coverage, completion time ($T_{95}$), and redundant multi-agent overlap in cooperative exploration?**

---

## Architecture & Methodology
- **Environment (`cce/envs/grid_world.py`):** Vectorised grid-world ($B$ parallel environments in NumPy) on a $20 \times 20$ map with $20\%$ obstacles, flood-fill connectivity guarantees, $N=4$ homogeneous agents, and partial observability via a local $5 \times 5$ FOV. Each agent's visited mask (`visited_local[b, i]`) is strictly private and updates only from its own $5 \times 5$ FOV, while `visited_global[b]` is used only by the centralised critic and evaluation metrics.
- **Algorithm (`cce/algo/mappo.py`):** CTDE MAPPO with parameter-shared actor $\pi_\theta(a_i^t, g_i^t \mid o_i^t)$ and centralised value critic $V_\phi(s^t)$ using GAE ($\gamma=0.99, \lambda_{\text{GAE}}=0.95$) and clipped PPO updates. Received messages update a separate belief map $\hat{\mathbf{B}}_i^t$ for quadrant deconfliction without overwriting `visited_local`.
- **Communication Bottleneck (`cce/comm/vq.py`):** Differentiable discrete bottleneck with an EMA-updated Vector Quantization (VQ) codebook ($K=16$ codes, 2 message slots, $d_m = 8$ bits per message) trained end-to-end with a Straight-Through Estimator (STE). Communication range is fixed at $R_{\text{comm}} = 3$ so $\beta$ directly controls per-step send frequency.
- **Two-Layer Budget Control (`cce/comm/budget.py`):**
  1. **Hard Cap (`TokenBucket`):** Allocates $\lfloor \beta N T \rfloor$ tokens per episode and masks send logits to $-\infty$ once tokens are exhausted (guaranteeing $\beta_{\text{actual}} \le \beta$, and $0$ bits at $\beta=0.0$).
  2. **Soft Penalty (KKT Dual $\lambda$):** Updates the non-negative Lagrangian multiplier $\lambda_{k+1} = \text{clip}(\lambda_k + \alpha_\lambda(\beta_{\text{actual},k} - \beta), 0, 5.0)$ and centres gate utility logits to transmit only high-utility messages.
- **Baselines (`cce/baselines/`):** Non-learning decentralized **Frontier** explorer (obstacle-aware nearest-frontier navigation over each agent's private local mask) and **Random** walk baseline.

---

## Experimental Results & Screenshots

All models and baselines were evaluated over **20 held-out episodes** (seed `42`, $20 \times 20$ grid, $N=4$ agents, $5 \times 5$ FOV, $T=100$ steps, $d_m=8$ bits/message).

### Quantitative Results Summary

| Method / Budget | Coverage Ratio $\uparrow$ | $T_{95}$ (Steps) $\downarrow$ | Success Rate $\uparrow$ | Overlap Ratio $\downarrow$ | Measured $\beta_{\text{actual}}$ | Bandwidth (Bits/Ep) | Dual $\lambda^*$ | Comm. Eff. $E_{\text{comm}}$ | Avg. Reward |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frontier (No Comm)** | `0.999 ± 0.003` | `24.6 ± 4.9` | `100.0%` | `0.633 ± 0.141` | `0.000` | `0.0` | — | `0.999` | `+1.46` |
| **Random (No Comm)** | `0.707 ± 0.081` | `100.0 ± 0.0` | `0.0%` | `0.418 ± 0.170` | `0.000` | `0.0` | — | `0.707` | `-2.98` |
| **MAPPO ($\beta=0.00$)** | `0.808 ± 0.064` | `100.0 ± 0.0` | `0.0%` | `0.541 ± 0.118` | `0.000` | `0.0` | `3.50` | **`0.808`** | `-1.12` |
| **MAPPO ($\beta=0.25$)** | `0.879 ± 0.042` | `99.5 ± 2.2` | `5.0%` | `0.568 ± 0.095` | `0.227` | `363.2` | `1.38` | `0.716` | `+0.45` |
| **MAPPO ($\beta=0.50$)** | `0.936 ± 0.031` | `80.8 ± 20.2` | `60.0%` | `0.512 ± 0.084` | `0.442` | `707.2` | `0.72` | `0.649` | `+3.18` |
| **MAPPO ($\beta=0.75$)** | **`0.972 ± 0.014`** | `52.0 ± 19.5` | **`90.0%`** | **`0.412 ± 0.076`** | `0.657` | `1051.2` | `0.28` | `0.587` | **`+5.42`** |
| **MAPPO ($\beta=1.00$)** | **`0.980 ± 0.039`** | **`49.7 ± 19.0`** | **`90.0%`** | `0.468 ± 0.089` | `0.872` | `1395.2` | `0.00` | `0.523` | `+5.31` |

### Key Takeaways
1. **Strict Two-Layer Budget Compliance:** Across all five target budgets $\beta \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$, measured communication $\beta_{\text{actual}} \in \{0.000, 0.227, 0.442, 0.657, 0.872\}$ stays strictly below $\beta$, with exact zero-bit enforcement at $\beta=0.0$.
2. **Monotonic Coverage Gain & Pareto Knee ($\beta^* = 0.75$):** Coverage rises from `0.808` ($0\%$ success) at $\beta=0.0$ to `0.972` ($90\%$ success, $T_{95}=52.0$ steps) at $\beta=0.75$. Increasing $\beta$ from $0.75$ to $1.00$ adds only `+0.8%` coverage for `+344` bits/episode, confirming $\beta^* = 0.75$ as the Kneedle knee point.
3. **34.9% Lower Multi-Agent Overlap than Frontier:** Because local visited masks are strictly private, uncoordinated Frontier agents suffer high multi-agent overlap (`0.633`) as multiple teammates sweep the same corridors. Communicating MAPPO agents ($\beta=0.75$) coordinate sector assignments, achieving `0.972` coverage while reducing redundant overlap to **`0.412`** (a **34.9% relative reduction**).

---

### Result Visualizations

| Document / Artifact | Path |
| :--- | :--- |
| **End-Term Final Report (8-Page PDF)** | [`reports/Team_37_Final_Report.pdf`](reports/Team_37_Final_Report.pdf) |
| **Mid-Term Project Idea Review (PDF)** | [`reports/Team_37_Project_Idea_Review.pdf`](reports/Team_37_Project_Idea_Review.pdf) |
| **Standardized Evaluation Output (JSON)** | [`results/eval_results.json`](results/eval_results.json) |
| **Kaggle Evaluation & Plotting Notebook** | [`notebooks/fai-project.ipynb`](notebooks/fai-project.ipynb) |

---

## Project Reports
- [Team 37 Project Idea Review](reports/Team_37_Project_Idea_Review.pdf) — Initial project proposal and literature review.
- [Team 37 Final Report](reports/Team_37_Final_Report.pdf) — Complete experimental results, analysis, and conclusions (to be added after final evaluation).

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

### 1. Verify with Unit Tests (37 Passing Tests)
Run the unit test suite across the grid-world environment, VQ bottleneck, two-layer budget controller, rollout buffer, and MAPPO trainer:
```bash
pytest tests/test_budget.py tests/test_buffer.py tests/test_grid_world.py tests/test_map_gen.py tests/test_mappo.py tests/test_vq.py -v
```

### 2. Evaluate Baselines & Trained MAPPO Checkpoints
```bash
# Evaluate Frontier exploration baseline (20 episodes, seed 42)
python scripts/evaluate.py --baseline frontier --output eval_frontier.json --grid_size 20 --num_agents 4 --fov_size 5 --max_steps 100 --num_episodes 20 --seed 42

# Evaluate Random exploration baseline (20 episodes, seed 42)
python scripts/evaluate.py --baseline random --output eval_random.json --grid_size 20 --num_agents 4 --fov_size 5 --max_steps 100 --num_episodes 20 --seed 42

# Evaluate a trained MAPPO checkpoint (e.g., β = 0.75)
python scripts/evaluate.py --baseline mappo --run_dir runs/budget_sweep/beta_0_75 --checkpoint runs/budget_sweep/beta_0_75/checkpoints/best_model.pt --output eval_mappo_beta_0_75.json --grid_size 20 --num_agents 4 --fov_size 5 --max_steps 100 --num_episodes 20 --seed 42 --dm_bits 8
```

### 3. Training MAPPO Models Across Budgets $\beta \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$
```bash
# Quick smoke test (2-5 min on CPU)
python train.py --run_dir runs/smoke --num_envs 16 --num_updates 50 --max_steps 32

# Full 300-update training run for a target budget (e.g., β = 0.75)
python train.py --beta 0.75 --run_dir runs/budget_sweep/beta_0_75 --num_envs 32 --num_updates 300 --max_steps 100 --grid_size 20 --num_agents 4 --fov_size 5
```

### 4. Interactive Web Dashboard (`app.py`)
Launch the trajectory replay viewer and multi-run Pareto comparison dashboard:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Project Structure
```text
cce/                          # Main package (communication-constrained exploration)
├── envs/                     # Vectorised grid-world (maps, FOV, connectivity, private masks)
├── comm/                     # Encoder, VQ-STE codebook, channel, TokenBucket + KKT budget controller
├── algo/                     # MAPPO trainer, rollout buffer, actor-critic networks
├── baselines/                # Non-learning Frontier and Random exploration baselines
├── eval/                     # Metrics (Cov, T95, Overlap), Pareto frontier + Kneedle analysis
└── utils/                    # Configs, logging, seeding
app.py                        # Streamlit interactive web dashboard
train.py                      # Main MAPPO training entry point
analyse.py                    # Multi-run Pareto and protocol analysis
scripts/
└── evaluate.py               # Standardised post-training evaluation CLI
configs/                      # Experiment YAML configs (Hydra)
tests/                        # Pytest unit test suite (37 tests)
```
```
