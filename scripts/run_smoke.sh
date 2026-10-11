#!/bin/bash
set -e

echo "=== FAI Project Smoke Test ==="
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo



# Check Python environment
echo "1. Checking Python environment..."
python3 --version
echo

# Check required dependencies
echo "2. Checking dependencies..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
echo

# Run VQ module self-test
echo "3. Running VQ module self-test..."
cd /Users/srimaan/.openclaw/workspaces/repo-cce
python3 cce/comm/vq.py
echo

# Check if tests directory exists
if [ -d "tests" ]; then
    echo "4. Running pytest..."
    python3 -m pytest tests/ -v
else
    echo "4. No tests directory found. Skipping pytest."
fi

echo
echo "=== Smoke Test Complete ==="
