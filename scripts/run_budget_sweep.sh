#!/usr/bin/env bash
# Run the full β-sweep across communication budgets.

set -e

cd "$(dirname "$0")/.."

BETAS="0.0 0.1 0.25 0.5 0.75 1.0"
SEEDS="0 1 2 3 4"
NUM_UPDATES=500
NUM_ENVS=64

echo "=== Budget sweep: β ∈ {$BETAS}, seeds ∈ {$SEEDS} ==="

for beta in $BETAS; do
    for seed in $SEEDS; do
        run_name="beta_${beta}_seed_${seed}"
        run_dir="runs/budget_sweep/${run_name}"
        
        echo ""
        echo "--- Training β=$beta seed=$seed ---"
        python3 train.py \
            --beta "$beta" \
            --seed "$seed" \
            --num_envs "$NUM_ENVS" \
            --num_updates "$NUM_UPDATES" \
            --run_dir "$run_dir" \
            --run_name "$run_name" \
            --grid_size 20 \
            --num_agents 4 \
            --fov_size 5 \
            --max_steps 128 \
            --actor_hidden 64 \
            --critic_hidden 256 \
            --n_codes 16 \
            --code_dim 8 \
            --n_slots 4 \
            --batch_size 3200 \
            --num_epochs 10

        echo "--- Evaluating β=$beta seed=$seed ---"
        python3 evaluate.py \
            --run_dir "$run_dir" \
            --num_episodes 50 \
            --num_envs 16 \
            --seed 42
    done
done

echo ""
echo "=== All training complete. Running analysis... ==="
python3 analyse.py \
    --results_dir runs/budget_sweep \
    --output_dir analysis

echo ""
echo "Budget sweep complete. Results in analysis/"
