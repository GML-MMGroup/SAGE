#!/bin/bash

# ==============================================================================
# EgoBench Competition - Evaluation Script
# ==============================================================================
#
# This script evaluates interaction results and generates evaluation reports for
# the competition.
#
# Usage:
#   bash run_eval.sh
#       Evaluate every result directory under results/ that contains JSON files.
#
# Optional: Specify model name and/or number of samples
#   bash run_eval.sh --model_name Qwen3.5-397B-A17B --num_samples 10
#   bash run_eval.sh --num_samples 10
# ==============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to analysis scripts directory because evaluate_interaction.py uses
# paths relative to this directory.
cd "$ANALYSIS_DIR"

# Default values
MODEL_NAME=""
NUM_SAMPLES=0
MODEL_NAME_PROVIDED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --model_name requires a value."
                exit 1
            fi
            MODEL_NAME="$2"
            MODEL_NAME_PROVIDED=true
            shift 2
            ;;
        --num_samples)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --num_samples requires a value."
                exit 1
            fi
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash run_eval.sh [--model_name MODEL_NAME] [--num_samples N]"
            echo ""
            echo "Without --model_name, evaluate all result directories under results/."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

RESULTS_ROOT="$PROJECT_ROOT/results"
NEW_RESULT_GLOB="[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_*.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_result_dir_eval() {
    local results_dir="$1"
    local model_name="${results_dir#"$RESULTS_ROOT"/}"
    local json_count

    json_count=$(find "$results_dir" -maxdepth 1 -type f -name "$NEW_RESULT_GLOB" | wc -l)
    if [[ "$json_count" -eq 0 ]]; then
        echo "Skipping '$model_name': no new-format JSON files found in '$results_dir'."
        return 0
    fi

    echo "------------------------------------------"
    echo "Model: $model_name"
    echo "Found $json_count result files."
    echo "Starting evaluation..."
    echo ""

    "$PYTHON_BIN" evaluate_interaction.py --model_name "$model_name" --num_samples "$NUM_SAMPLES"
}

run_model_eval() {
    local model_name="$1"
    local results_dir="$RESULTS_ROOT/$model_name"
    local result_dirs

    if [[ ! -d "$results_dir" ]]; then
        echo "Error: Results directory '$results_dir' does not exist."
        return 1
    fi

    mapfile -t result_dirs < <(find "$results_dir" -type f -name "$NEW_RESULT_GLOB" -printf "%h\n" | sort -u)
    if [[ "${#result_dirs[@]}" -eq 0 ]]; then
        echo "Skipping '$model_name': no new-format JSON files found under '$results_dir'."
        return 0
    fi

    for result_dir in "${result_dirs[@]}"; do
        run_result_dir_eval "$result_dir"
        echo ""
    done
}

echo "=========================================="
echo "EgoBench Competition - Evaluation"
echo "=========================================="
echo ""
echo "Configuration:"
if [[ "$MODEL_NAME_PROVIDED" == true ]]; then
    echo "  Model Name: $MODEL_NAME"
else
    echo "  Model Name: all result directories"
fi
echo "  Num Samples: $NUM_SAMPLES"
echo ""

if [[ "$MODEL_NAME_PROVIDED" == true ]]; then
    run_model_eval "$MODEL_NAME"
else
    if [[ ! -d "$RESULTS_ROOT" ]]; then
        echo "Error: Results root '$RESULTS_ROOT' does not exist."
        echo "Please run the simulations first using: bash run_all_scenarios.sh"
        exit 1
    fi

    mapfile -t RESULT_DIRS < <(find "$RESULTS_ROOT" -type f -name "$NEW_RESULT_GLOB" -printf "%h\n" | sort -u)

    if [[ "${#RESULT_DIRS[@]}" -eq 0 ]]; then
        echo "Error: No new-format JSON result files found under '$RESULTS_ROOT'."
        echo "Please run simulations that write files like: YYYYMMDD_HHMMSS_retail1_easy.json"
        exit 1
    fi

    echo "Found ${#RESULT_DIRS[@]} result directories."
    echo ""

    for results_dir in "${RESULT_DIRS[@]}"; do
        run_result_dir_eval "$results_dir"
        echo ""
    done
fi

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
echo ""
echo "Evaluation results saved under eval_result/ with the same relative structure as results/."
echo ""
echo "To view detailed results, check the corresponding eval_result directory."
