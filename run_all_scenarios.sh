#!/bin/bash

# ==============================================================================
# EgoBench Track 2 - Run Scenarios
# ==============================================================================
#
# Usage:
#   bash run_all_scenarios.sh
#   bash run_all_scenarios.sh --final_eval
#   bash run_all_scenarios.sh --final_eval --num_tasks 5
#
# Optional environment overrides:
#   SERVICE_MODEL_NAME, USER_MODEL_NAME, VIDEO_MODE
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_TASKS=0
FINAL_EVAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --final_eval)
            FINAL_EVAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -f ".env" ]; then
    set -a
    source ".env"
    set +a
fi

: "${USER_MODEL_NAME:=Qwen3.5-397B-A17B}"
: "${SERVICE_MODEL_NAME:=Qwen3.5-397B-A17B}"
: "${VIDEO_MODE:=local}"

export USER_MODEL_NAME SERVICE_MODEL_NAME VIDEO_MODE

echo "=========================================="
echo "EgoBench Track 2 - Running Scenarios"
echo "=========================================="
echo "User Model:       $USER_MODEL_NAME"
echo "Service Model:    $SERVICE_MODEL_NAME"
echo "Video Mode:       $VIDEO_MODE"
echo "Visual Input:     image_base64"
echo "Final Eval:       $FINAL_EVAL"
echo "Num Tasks:        $NUM_TASKS"
echo ""

mkdir -p "results/${SERVICE_MODEL_NAME}"

run_scenario() {
    local scenario="$1"
    local scenario_number="$2"

    echo "Running: ${scenario}${scenario_number} (easy mode)"
    python run/multi_agent.py \
        --scenario "$scenario" \
        --scenario_number "$scenario_number" \
        --service_model_name "$SERVICE_MODEL_NAME" \
        --multi_agent_user \
        --num_tasks "$NUM_TASKS"
    echo "Completed: ${scenario}${scenario_number}"
    echo ""
}

if [ "$FINAL_EVAL" = true ]; then
    run_scenario "retail" 6
    run_scenario "retail" 10
    run_scenario "kitchen" 4
    run_scenario "restaurant" 5
    run_scenario "order" 2
else
    for i in $(seq 1 10); do
        run_scenario "retail" "$i"
    done
    for i in $(seq 1 4); do
        run_scenario "kitchen" "$i"
    done
    for i in $(seq 1 5); do
        run_scenario "restaurant" "$i"
    done
    for i in $(seq 1 2); do
        run_scenario "order" "$i"
    done
fi

echo "=========================================="
echo "All scenarios completed."
echo "Results saved under: results/${SERVICE_MODEL_NAME}/image_base64/"
echo "=========================================="
