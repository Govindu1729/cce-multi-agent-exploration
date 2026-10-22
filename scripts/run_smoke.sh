#!/usr/bin/env bash
# Smoke test: quick run to verify the code works.

set -e

cd "$(dirname "$0")/.."

echo "=== Smoke test ==="
python3 train.py \
    --num_envs 16 \
    --num_updates 50 \
    --grid_size 10 \
    --num_agents 4 \
    --fov_size 3 \
    --max_steps 32 \
    --actor_hidden 32 \
    --critic_hidden 64 \
    --n_codes 8 \
    --code_dim 4 \
    --n_slots 2 \
    --batch_size 1600 \
    --num_epochs 5 \
    --beta 0.5 \
    --run_dir runs/smoke \
    --run_name smoke_test \
    --seed 0

echo ""
echo "=== Evaluating smoke test ==="
python3 evaluate.py \
    --run_dir runs/smoke \
    --num_episodes 10 \
    --num_envs 8 \
    --seed 42

echo ""
echo "Smoke test complete. Results in runs/smoke/eval_results.json"
