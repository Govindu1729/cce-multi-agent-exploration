# Streamlit Web UI Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Streamlit web UI layer (`app.py`) into a zero-CLI, interactive multi-agent trajectory replay and multi-run Pareto comparison dashboard while preserving 100% backward compatibility with all existing evaluation and recorder code.

**Architecture:** Approach B: Self-contained, robust web UI layer in `app.py`. Implements cached loaders (`@st.cache_data`) for `.pkl`, `.json`, and `.csv` files, an interactive Plotly grid replay with FOV overlays and Manhattan-distance communication links, synchronized per-frame metrics with $T_{95}$ reference lines, playback controls, and a dedicated multi-run Pareto frontier comparison tab.

**Tech Stack:** Python 3.10+, Streamlit (>=1.30), Plotly (graph_objects), NumPy, Pandas, SciPy.

**Spec:** User request constraints: Zero modification of simulation logic, reward functions, policy inference, metric formulas, or output schema of `cce/eval/recorder.py` and `scripts/evaluate.py`. Do NOT commit or push to git.

## Global Constraints

- Do NOT modify `cce/eval/recorder.py` or `scripts/evaluate.py`.
- Do NOT alter any metric formulas or output schemas.
- Maintain 100% backward compatibility with any `.pkl` files created by `recorder.py` or `scripts/evaluate.py`.
- Do NOT commit anything to git.
- Use `@st.cache_data` on file loading and heavy metric computations.

## Review Focus

1. **Trajectory Frame Schema Compatibility:** Handle both dataclass `EpisodeStep` objects and raw dictionary objects seamlessly.
2. **Missing/Empty Run Handling:** Handle directories with no runs, empty pickle files, or runs with 0 communication gracefully without crashing.
3. **Multi-Run Format Flexibility:** Allow comparing multiple `.pkl` trajectory files, `.json` evaluation result files, and `.csv` summary tables in Tab 2.
4. **Grid Coordinate Inversion:** Ensure $(y, x)$ coordinate orientation matches between grid numpy arrays (row, col) and Plotly axes (y, x) with $y$-axis inverted to match matrix representation.
5. **Playback State Management:** Manage Streamlit session state (`st.session_state`) for playback step indexing and auto-advance cleanly across reruns.

---

### Task 1: Core Data Loaders & Multi-File Discovery

**Files:**
- Modify: `cce-multi-agent-exploration/app.py`

**Interfaces:**
- Consumes: Raw file bytes or file paths from `runs/`, `runs/viz/`, or uploaded files.
- Produces: Normalized trajectory structures, cached scan lists for `.pkl`, `.json`, `.csv`.

- [ ] **Step 1: Implement directory scanner & cached file loaders**
  Implement `scan_available_runs(base_dirs)` to discover `.pkl`, `.json`, and `.csv` files across `runs/`, `runs/viz/`, and the working directory.
  Implement `@st.cache_data` loaders: `load_trajectory_bytes`, `load_json_bytes`, `load_csv_bytes`, and normalized frame parser `get_frame`.
- [ ] **Step 2: Add sidebar file selector and multi-file uploader**
  Add multi-file uploader supporting `.pkl`, `.json`, `.csv`.
  Add dropdowns for selecting active run for Replay (Tab 1) and multi-select for Comparison (Tab 2).

---

### Task 2: Interactive Grid & Communication Replay Layer

**Files:**
- Modify: `cce-multi-agent-exploration/app.py`

**Interfaces:**
- Consumes: Single frame dict from loaded trajectory.
- Produces: Plotly `go.Figure` with obstacles, visited cells, cell overlap highlighting, agent markers, $k\times k$ FOV overlays, and communication links.

- [ ] **Step 1: Compute cumulative overlap and agent visited masks**
  Compute per-cell visit counts across agents from history up to step $t$ and highlight cells visited by $\ge 2$ agents.
- [ ] **Step 2: Implement enhanced `grid_figure`**
  - Layer 1: Obstacles (`grid`)
  - Layer 2: Visited cells colored by first-visitor agent
  - Layer 3: Overlap highlight layer for cells visited by multiple agents
  - Layer 4: $k\times k$ local FOV bounding boxes around each agent (toggleable, default $k=5$)
  - Layer 5: Potential communication links (Manhattan distance $\le 3$) in subtle dashed lines
  - Layer 6: Active communication links for `messages` sent in the current frame (bold colored lines with sender/receiver markers)
  - Layer 7: Agent position stars with distinct colors and hover tooltips
- [ ] **Step 3: Implement playback controls**
  Add Play/Pause button, Step Back (`⏮`), Step Forward (`⏭`), Reset (`|◀`), End (`▶|`), Playback Speed selectbox/slider, and Timeline scrubber slider.

---

### Task 3: Synchronized Metrics & Multi-Run Pareto Comparison Views

**Files:**
- Modify: `cce-multi-agent-exploration/app.py`

**Interfaces:**
- Consumes: Loaded trajectories, evaluation JSONs, evaluation CSVs.
- Produces: Tab 1 synchronized curves & Tab 2 multi-run comparison table and Pareto frontier plot.

- [ ] **Step 1: Build Tab 1 (Episode Replay)**
  Display synchronized metric cards: Coverage $Cov(t)$, Overlap fraction, Bandwidth bits (step & cumulative), active message count, comm efficiency.
  Display Coverage over time curve with the horizontal 95% target dashed line and current step marker.
  Display Cumulative Bandwidth curve over time.
  Display Overlap over time curve.
  Display current frame message table and full episode message expander.
- [ ] **Step 2: Build Tab 2 (Multi-Run & Pareto Comparison)**
  Extract metrics ($T_{95}$, Final Coverage, Overlap, Success Rate, Bandwidth, Comm Efficiency) from 2+ selected runs.
  Display side-by-side metric comparison table and comparative bar charts.
  Plot Bandwidth vs Coverage Pareto frontier using `cce.eval.pareto.pareto_frontier` and identify the Knee point via `kneedle_knee` when $\ge 3$ points.

---

### Task 4: Verification & Smoke Testing

**Files:**
- Verify: `cce-multi-agent-exploration/app.py`

- [ ] **Step 1: Syntax and import validation**
  Run Python syntax compilation and test script loading `app.py` in headless mode.
- [ ] **Step 2: Verify zero changes to restricted files**
  Confirm `cce/eval/recorder.py` and `scripts/evaluate.py` have zero modifications.
- [ ] **Step 3: Test loading existing trajectories**
  Verify loading `runs/viz/trajectory.pkl` and `eval_results.json` works cleanly and renders figures.
