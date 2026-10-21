# Project Status & Next Steps

## ✅ What's Done

**Complete implementation** of the Communication-Constrained Multi-Agent Exploration project:

### Core Components
- ✅ **Vectorised grid-world environment** (`cce/envs/grid_world.py`)
  - Pure NumPy, B parallel envs, fixed-length episodes
  - Procedural map generation with connectivity guarantees
  - Local visited masks (no global info leakage)
  - Manhattan-distance communication (R=3)
  
- ✅ **MAPPO with communication** (`cce/algo/mappo.py`)
  - Shared actor + centralised critic (CNN)
  - GAE, value normalisation, PPO clipping
  - Supports MLP and GRU backbones
  
- ✅ **VQ bottleneck** (`cce/comm/vq.py`)
  - EMA codebook updates, dead-code revival
  - Straight-through gradient estimator
  - Configurable: K codes, d_code dim, d_m slots
  
- ✅ **Budget controller** (`cce/comm/budget.py`)
  - Lagrangian dual with PID controller
  - Policy mode (logit bias) or utility mode (threshold gate)
  - Exact β enforcement via per-episode token budget
  
- ✅ **Evaluation pipeline** (`cce/eval/`)
  - Coverage, T95, overlap, success rate, bandwidth, efficiency
  - Pareto frontier with Kneedle knee detection
  - Message entropy and mutual information analysis
  
- ✅ **Training scripts** (`train.py`, `evaluate.py`, `analyse.py`)
- ✅ **Configs** (Hydra YAML for all experiments)
- ✅ **Tests** (pytest for every module)
- ✅ **Documentation** (README, QUICKSTART, this file)

---

## 🚀 What to Do Next

### Step 1: Verify Installation (5 min)
```bash
cd /Users/srimaan/Sem7/FAI/Project
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest tests/ -v
```

### Step 2: Run Smoke Test (5 min)
```bash
bash scripts/run_smoke.sh
```
Check `runs/smoke/eval_results.json` — should have coverage > 0.3.

### Step 3: Short Local Training (30 min)
```bash
python train.py \
    --num_envs 32 \
    --num_updates 100 \
    --grid_size 20 \
    --num_agents 4 \
    --beta 0.25 \
    --run_dir runs/local_test \
    --use_wandb \
    --run_name local_test
```
Watch W&B for learning curves. Coverage should reach ~0.7-0.9.

### Step 4: Full β-Sweep on Kaggle (2-4 hours)
1. Zip the project:
   ```bash
   cd /Users/srimaan/Sem7/FAI/Project
   zip -r cce_project.zip . -x "*.pyc" "__pycache__/*" ".venv/*"
   ```
2. Upload to Kaggle as dataset
3. Create notebook, install:
   ```python
   !pip install -e /kaggle/input/cce_project/cce_project
   ```
4. Run:
   ```python
   !cd /kaggle/input/cce_project/cce_project && bash scripts/run_budget_sweep.sh
   ```
5. Download results from `/kaggle/working/`

### Step 5: Analyse Results
```bash
python analyse.py \
    --results_dir runs/budget_sweep \
    --output_dir analysis
```
Generates `analysis/pareto_frontier.png`, `analysis/entropy_vs_beta.png`, etc.

---

## 📊 Expected Results

### Coverage vs β
- β=0 (silence): coverage ~0.5-0.7
- β=0.1: coverage ~0.7-0.8
- β=0.25: coverage ~0.85-0.95
- β=0.5: coverage ~0.95+
- β=1.0: coverage ~0.95+ (diminishing returns)

### Pareto Frontier
The knee (optimal β) should be around **β=0.25-0.5** — this is where you get most of the coverage benefit with moderate bandwidth.

### Message Entropy
- High β: entropy ~3.5-4 bits (all codes used equally)
- Low β: entropy ~1-2 bits (only a few codes used)

### T95
- β=0: T95 ~ 100-128 steps (slow, no coordination)
- β=0.5: T95 ~ 50-80 steps (fast, good coordination)

---

## 🔧 Potential Issues & Fixes

### 1. Low Coverage (< 0.5)
**Cause**: Agents not exploring effectively.
**Fix**: Increase `--entropy_coef` (0.05 instead of 0.01) or reduce `--w_time` (0.005 instead of 0.01).

### 2. Budget Not Respected
**Cause**: PID controller not converged.
**Fix**: Increase `--num_updates` (500+). Or tune PID: `pid_kp=1.0, pid_ki=0.1`.

### 3. Codebook Collapse (perplexity < 2)
**Cause**: VQ not using all codes.
**Fix**: Reduce `--n_codes` (8 instead of 16) or increase `--entropy_coef`.

### 4. CUDA OOM
**Cause**: Too many parallel envs or large batch.
**Fix**: Reduce `--num_envs` (32 instead of 64) or `--batch_size` (1600 instead of 3200).

### 5. Slow Training
**Cause**: Large grid or many agents.
**Fix**: Use `--critic_type mlp` instead of `cnn` for small grids (20×20).

---

## 📝 Paper Writing Checklist

### Figures to Generate
- [ ] **Fig 1**: Environment visualisation (grid with agents, FOV, visited cells)
- [ ] **Fig 2**: Coverage vs β (line plot with error bars)
- [ ] **Fig 3**: T95 vs β
- [ ] **Fig 4**: Pareto frontier (bandwidth vs coverage) with knee marked
- [ ] **Fig 5**: Message entropy vs β
- [ ] **Fig 6**: Codebook usage heatmap (which codes are used when)
- [ ] **Fig 7**: Example trajectories for β=0 vs β=0.5 (show coordination)

### Tables to Generate
- [ ] **Table 1**: Main results (coverage, T95, success rate, bandwidth) for β ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}
- [ ] **Table 2**: Agent scaling (N=3, 4, 5 at β=0.25)
- [ ] **Table 3**: Grid scaling (G=20, 30, 50 at β=0.25)
- [ ] **Table 4**: Ablation (no comm, no budget, no VQ, no GRU)

### Analysis to Run
- [ ] **Kneedle**: identify optimal β
- [ ] **Message entropy**: how diverse are messages?
- [ ] **Mutual information**: what do messages encode? (coverage, distance, etc.)
- [ ] **Qualitative**: visualise policies for β=0 vs β=0.5 (do agents coordinate?)

---

## 🎓 Viva / Presentation Prep

### Key Points to Emphasise
1. **Exact β enforcement** via Lagrangian dual (not just a penalty)
2. **Differentiable discrete communication** via VQ + ST estimator
3. **Scalable environment** (pure NumPy, no IPC overhead)
4. **Reproducible** (seeds, deterministic env generation, fixed-length episodes)
5. **Comprehensive evaluation** (Pareto, Kneedle, message analysis)

### Potential Questions
- **Q**: Why MAPPO instead of QMIX/MADDPG?
  **A**: MAPPO is on-policy, handles partial observability well, and is simpler to implement with a communication bottleneck. QMIX would require modifying the mixing network to incorporate messages.

- **Q**: Why not use a continuous message space?
  **A**: Discrete messages better model real-world bandwidth constraints (bits), and VQ provides a differentiable approximation that allows end-to-end training.

- **Q**: How do you know agents are actually communicating useful information?
  **A**: We compute mutual information between messages and state features (coverage, distance). High MI indicates meaningful communication. Also, coverage improves with β, showing communication helps.

- **Q**: What happens if β is too low?
  **A**: Coverage drops, T95 increases, and agents explore independently (no coordination). The Pareto frontier shows the trade-off.

- **Q**: Why is the knee at β=0.25-0.5?
  **A**: Diminishing returns — beyond β=0.5, agents already share enough info for near-optimal coordination. Extra bandwidth doesn't help much.

---

## 📚 References (for paper)

1. Yu et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS.
2. Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
3. Oliehoek & Amato (2016). "A Concise Introduction to Decentralized POMDPs." Springer.
4. Liu et al. (2022). "Adaptive Discrete Communication Bottlenecks with Dynamic Vector Quantization." arXiv:2202.01334.
5. Freed et al. (2020). "Communication Learning via Backpropagation in Discrete Channels." AAAI.
6. Satopaa et al. (2011). "Finding a 'Kneedle' in a Haystack." ICDCSW.

---

## 🙏 Acknowledgments

This project was built with extensive AI assistance (ChatGPT/Claude) for code generation, debugging, and architectural design. All algorithmic decisions were made collaboratively with the human author (Govindu Srimaan).

---

**Status**: ✅ Implementation complete, ready for training and evaluation.
**Next**: Run smoke test → short local training → full β-sweep on Kaggle → analyse results → write paper.
