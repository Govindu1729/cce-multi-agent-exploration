#!/bin/bash
# =============================================================================
# Decoupled Budget Sweeps Runner
# =============================================================================
# This script runs the three decoupled budget sweep experiments:
# 1. budget_dm_sweep.yaml: Fix frequency=1, topology=fully_connected, sweep dm ∈ {1,2,4,8,16}
# 2. budget_fc_sweep.yaml: Fix dm=4, topology=fully_connected, sweep frequency ∈ {0.1,0.25,0.5,1.0}
# 3. budget_topology_sweep.yaml: Fix dm=4, frequency=1, sweep topology types
#
# Usage: ./scripts/run_decoupled_sweeps.sh [sweep_name]
#   sweep_name: one of "dm", "frequency", "topology", or "all" (default)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default to running all sweeps
SWEEP_TYPE="${1:-all}"

# Project root
PROJECT_ROOT="/Users/srimaan/.openclaw/workspaces/repo-cce"
CONFIGS_DIR="${PROJECT_ROOT}/configs/exp"
VENV="${PROJECT_ROOT}/venv"

# Activate virtual environment
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
    echo -e "${GREEN}Activated virtual environment${NC}"
else
    echo -e "${RED}Virtual environment not found at ${VENV}${NC}"
    exit 1
fi

# Check if hydra is available
if ! python -c "import hydra" 2>/dev/null; then
    echo -e "${YELLOW}Installing hydra-core...${NC}"
    pip install hydra-core
fi

# Function to run a sweep
run_sweep() {
    local config_file="$1"
    local sweep_name="$2"
    
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Running ${sweep_name} sweep${NC}"
    echo -e "${BLUE}Config: ${config_file}${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    
    cd "${PROJECT_ROOT}"
    
    # Run with hydra multirun
    python -m cce.train \
        --config-dir "${CONFIGS_DIR}" \
        --config-name "$(basename "${config_file}" .yaml)" \
        hydra.mode=MULTIRUN \
        2>&1 | tee "logs/${sweep_name}_sweep_$(date +%Y%m%d_%H%M%S).log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "\n${GREEN}✓ ${sweep_name} sweep completed successfully${NC}"
    else
        echo -e "\n${RED}✗ ${sweep_name} sweep failed${NC}"
        return 1
    fi
}

# Create logs directory
mkdir -p "${PROJECT_ROOT}/logs"

# Run requested sweeps
case "${SWEEP_TYPE}" in
    "dm")
        run_sweep "${CONFIGS_DIR}/budget_dm_sweep.yaml" "dm"
        ;;
    "frequency"|"freq"|"fc")
        run_sweep "${CONFIGS_DIR}/budget_fc_sweep.yaml" "frequency"
        ;;
    "topology"|"topo")
        run_sweep "${CONFIGS_DIR}/budget_topology_sweep.yaml" "topology"
        ;;
    "all")
        echo -e "${BLUE}Running all three decoupled budget sweeps...${NC}"
        
        # Run dm sweep
        run_sweep "${CONFIGS_DIR}/budget_dm_sweep.yaml" "dm"
        
        # Run frequency sweep
        run_sweep "${CONFIGS_DIR}/budget_fc_sweep.yaml" "frequency"
        
        # Run topology sweep
        run_sweep "${CONFIGS_DIR}/budget_topology_sweep.yaml" "topology"
        
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}All sweeps completed!${NC}"
        echo -e "${GREEN}========================================${NC}"
        ;;
    *)
        echo -e "${RED}Unknown sweep type: ${SWEEP_TYPE}${NC}"
        echo "Usage: $0 [dm|frequency|topology|all]"
        exit 1
        ;;
esac

echo -e "\n${BLUE}Results available in:${NC}"
echo -e "  ${PROJECT_ROOT}/outputs/"
echo -e "  ${PROJECT_ROOT}/multirun/"
echo -e "\n${BLUE}Logs available in:${NC}"
echo -e "  ${PROJECT_ROOT}/logs/"