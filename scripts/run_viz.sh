#!/bin/bash
# =============================================================================
# Run Visualization Pipeline
# =============================================================================
# This script:
# 1. Runs one episode with trajectory recording
# 2. Launches the Streamlit dashboard
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="/Users/srimaan/.openclaw/workspaces/repo-cce"
VENV="${PROJECT_ROOT}/venv"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FAI Visualization Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

# Activate venv
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}Virtual environment not found, using system Python${NC}"
fi

cd "${PROJECT_ROOT}"

# Step 1: Record trajectory if not exists
TRAJECTORY_PATH="runs/viz/trajectory.pkl"
if [ ! -f "${TRAJECTORY_PATH}" ]; then
    echo -e "\n${BLUE}Step 1: Recording trajectory...${NC}"
    python scripts/evaluate.py \
        --baseline frontier \
        --record \
        --record-output "${TRAJECTORY_PATH}" \
        --num-episodes 1 \
        --seed 42 \
        --grid-size 10 \
        --num-agents 4 \
        --fov-radius 2 \
        --max-steps 50 \
        --comm-frequency 1.0 \
        --topology fully_connected \
        --comm-budget 10 \
        --max-dm 4
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Trajectory recorded to ${TRAJECTORY_PATH}${NC}"
    else
        echo -e "${RED}✗ Failed to record trajectory${NC}"
        exit 1
    fi
else
    echo -e "\n${BLUE}Step 1: Trajectory already exists at ${TRAJECTORY_PATH}${NC}"
fi

# Step 2: Launch Streamlit
echo -e "\n${BLUE}Step 2: Launching Streamlit dashboard...${NC}"
echo -e "${YELLOW}Opening http://localhost:8501 in your browser...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}\n"

streamlit run app.py --server.port 8501 --server.headless true