#!/usr/bin/env bash
# Setup script for Communication-Constrained Multi-Agent Exploration project.

set -e

echo "=== Setting up CCE project ==="

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.9+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install project in editable mode with dev dependencies
echo "Installing project dependencies..."
pip install -e ".[dev]"

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""
echo "To run smoke test:"
echo "  bash scripts/run_smoke.sh"
