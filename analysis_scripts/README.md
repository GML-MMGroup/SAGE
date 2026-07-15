# Analysis Scripts

This directory contains all Python scripts used for analyzing evaluation results.

## Script List

### Core Evaluation Scripts

1. **evaluate_interaction.py** - Interaction success rate evaluation script
   - Evaluates model tool calls and result correctness in customer service scenarios
   - Usage: `python evaluate_interaction.py --model_name <model_name>`

2. **print_eval.py** - Evaluation result printing script
   - Prints detailed evaluation results and statistics

### Analysis Scripts

3. **analyze_error_reasons.py** - Error analysis script
   - Analyzes error types and distributions across models
   - Counts tool call errors, execution errors, etc.

4. **analyze_order.py** - Order scenario restaurant selection analysis
   - Analyzes whether each model selected the correct restaurant in order scenarios

5. **calc_user_perf.py** - User performance metrics calculation script
   - Calculates performance metrics for the user agent

6. **count_stats.py** - Statistics script
   - Counts tool numbers, item numbers, scenario files, etc.

### Visualization Scripts

7. **plot_results.py** - Result visualization script
   - Generates academic-style statistical charts
   - Outputs bar charts, donut charts, scatter plots, LaTeX tables, etc.

## Path Configuration

All scripts are configured to run from the `analysis_scripts` directory, with relative paths set to:
- `../results` - Evaluation result directory
- `../eval_result` - Evaluation output directory
- `../scenarios` - Scenario file directory
- `../tools` - Tool directory

## Usage

Run scripts from the `analysis_scripts` directory or the project root:

```bash
# From project root
cd /ossfs/workspace/process_data
python analysis_scripts/evaluate_interaction.py --model_name glm-4.5v

# Or from the analysis_scripts directory
cd /ossfs/workspace/process_data/analysis_scripts
python evaluate_interaction.py --model_name glm-4.5v
```

## Dependencies

Ensure the following dependencies are installed:
- pandas
- numpy
- matplotlib
- Pillow
- adjustText
- argparse (Python standard library)