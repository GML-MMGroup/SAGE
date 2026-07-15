# SAGE: Execution-Grounded Adaptive Planning for Egocentric Interactive Agents

[Quick Start](#quick-start) · [Reproducing the Experiments](#reproducing-the-final-five-scenarios)

SAGE (**S**tatus-**A**ware **G**rounded **E**xecution) is designed for egocentric interactive agents: the system must identify the correct entities from videos and multi-constraint user requests, invoke structured tools, and ensure that the final database state is consistent with the user's intent. SAGE maintains visual evidence, tool results, step states, and unresolved constraints within a single execution loop. It commits to a branch action only after sufficient evidence has been gathered and uses the Reporter to perform targeted repairs on failed steps.

This repository provides the current reproduction and extended implementation for EgoBench Track 2, including multi-agent interaction, temporal frame selection, generic target localization, independent bounding-box review, visual recognition constrained by a candidate catalog, Planner–Executor–Reporter tool execution, and per-stage latency and token statistics.

## Overview

Egocentric interaction involves visual reference resolution, latent attributes, conditional branches, and state updates. Common paradigms exhibit different shortcomings in this setting: ReAct may commit to a branch prematurely based on the latest local observation, while the fixed natural-language plan used by Plan-and-Solve may execute mutually exclusive branches sequentially. SAGE instead uses explicit step states and execution evidence to decide whether to continue, repair, or terminate.

![Comparison of the ReAct, Plan-and-Solve, and SAGE paradigms](https://huggingface.co/datasets/thomasrios/TYPERA/resolve/main/SAGE/SAGE_000.png)

SAGE's core design includes:

- **Status-aware planning**: Steps have `pending`, `solved`, and `unresolved` states and are completed only after the evidence and state transition have been verified.
- **Probe–execution separation**: The system first obtains the evidence needed to evaluate a condition, then executes only the branch supported by that evidence.
- **Evidence compression**: The Planner uses a compact tool view, the Executor receives only the schema relevant to the current step, and the Reporter compresses raw tool traces into verified facts and unresolved requirements.
- **Visual–tool decoupling**: The visual branch produces normalized Visual Facts. If the tool branch discovers a missing visual entity, it can issue a type-constrained visual request and then resume from the same state.
- **Conservative visual fallback**: When localization is uncertain, a bounding box is invalid, or the Reviewer rejects a candidate, the entire batch of localization results is discarded and the system falls back to the original evidence.

## System Architecture

![Overall SAGE architecture](https://huggingface.co/datasets/thomasrios/TYPERA/resolve/main/SAGE/SAGE_001.png)

SAGE consists of Human–Agent Interaction, a Visual Recognition Module, and a Tool-Execution System. Its main components are:

| Component       | Responsibility in the Current Implementation                 |
| --------------- | ------------------------------------------------------------ |
| Simulated User  | Initiates a multi-turn conversation based on the scenario task instructions. |
| User Corrector  | Checks and corrects off-task content before simulated-user replies reach the service side. |
| Supervisor      | Routes requests to visual recognition, tool execution, or `ask_user`. |
| Frame Selector  | Selects task-relevant primary frames, auxiliary frames, and the target count along the timeline. |
| Box Locator     | Produces pixel-space candidate boxes with confidence states on the primary frames. |
| Frames Reviewer | Accepts, corrects, or rejects candidate boxes; any rejection triggers a fallback for the entire batch. |
| Visual Agent    | Identifies normalized entities using crops, full boxed frames, and auxiliary evidence. |
| Planner         | Generates the plan for the current round from the current state and outputs the `requires_next_planning_round` control field. |
| Executor        | Executes the plan for the current round; results accumulate and are passed to subsequent Planning Rounds. |
| Reporter        | Verifies accumulated results at the end of a Planning Round, generates a response, and returns any unresolved requirements. |

The current execution loop can be summarized as follows:

```text
User request
    │
    ▼
Supervisor ──────► ask_user
    │
    ├──► Visual branch ──► normalized Visual Facts ──┐
    │                                                │
    └──► Tool branch ─────────────────────────────────┤
                                                      ▼
                                        Planner (Round r)
                                              │
                                              │ plan + requires_next_planning_round
                                              ▼
                                           Executor
                                              │
                                              │ accumulated Tool Results
                         ┌────────────────────┴────────────────────┐
                         │                                         │
       requires_next_planning_round = true       false or round limit reached
                         │                                         │
                         ▼                                         ▼
               Planner (Round r + 1)                           Reporter
               + Previous Tool Results                            │
               + Previous Tool Descriptions                       │ unresolved?
                         │                                         │
                         └──────► repeat Planning Round             ├── no ──► verified response/state
                                                                   │
                                                                   └── yes
                                                                        ▼
                                                 Repair Planner ──► Executor ──► Reporter
                                                        ▲                            │
                                                        └── unresolved (max 2) ──────┤
                                                                                     └── resolved
                                                                                           ▼
                                                                                 verified response/state
```

The main Planning Round contains only the Planner and Executor. When the Planner sets `requires_next_planning_round` to `true`, the Executor completes the current round, after which the system appends the accumulated Tool Results and descriptions of the tools used to the next Planner input. This loop runs for at most 5 rounds. The Reporter is called only after the Planner sets the field to `false` or the round limit is reached. If the Reporter still returns `unresolved`, the system enters a separate repair-planning process and performs at most 2 additional rounds of Planner–Executor–Reporter repair.

## Temporal Visual Grounding and Paired Evidence

![SPG paired visual evidence](https://huggingface.co/datasets/thomasrios/TYPERA/resolve/main/SAGE/SAGE_002.png)

SAGE handles “which target the user is referring to” separately from “what the target actually is.” The system first resolves pointing references, temporal order, and action relationships along the timeline to select the primary evidence that identifies the target. When the target is occluded or its label is unclear, the system then selects auxiliary evidence that shows the same entity. Primary evidence answers **which target**, while complementary evidence supplies **what identity**.

The visual branch uses a `select → ground → review → render → recognize → normalize` pipeline:

1. Frames are sampled from the video at 1-second intervals. Based on the task, temporal order, and historical Visual Facts, the Frame Selector chooses primary frames, any necessary auxiliary frames, and the target count.
2. The Box Locator localizes targets only in the primary frames and returns pixel-space bounding boxes, evidence types, rationales, and `confident` / `uncertain` states.
3. Green `Target` boxes are rendered only when every candidate in the batch is reliable and geometrically valid. Otherwise, the system immediately falls back to unboxed primary and auxiliary frames.
4. The Frames Reviewer returns `accept`, `correct`, or `reject` for each box in the context of the full frame. Any rejection or invalid correction invalidates the entire batch of boxes.
5. Once all candidates pass review, the system organizes the Recognition VLM input as “target crop, full primary frame with a green box, and unboxed auxiliary frames.”
6. Recognition results must be matched back to the scene's candidate catalog after it has been filtered by the expected key. Unmatched results and free-form text referring to items outside the database are not passed to tool calls.

The green boxes are high-confidence candidate evidence that has passed localization and review, not irrefutable ground truth. The Recognition VLM must still validate them against the original task, full frames, and auxiliary evidence. Auxiliary frames also cannot replace the object identified by the anchor with a clearer but unrelated neighboring target.

## Experimental Results

### Evaluation Protocol

The final evaluation covers `retail6`, `retail10`, `order2`, `kitchen4`, and `restaurant5`. The main metrics are:

| Metric         | Meaning                                                      |
| -------------- | ------------------------------------------------------------ |
| Tool Success   | Whether all required tool names and arguments fully match the ground truth. |
| Result Success | Whether the database state after tool execution matches the reference state. |
| Joint Success  | Whether both Tool Success and Result Success are achieved.   |
| Tokens         | The average total number of input and output tokens per task. |
| Time           | The average per-task duration used by the corresponding experiment record; see the notes below each table for differences in timing definitions. |

### Base and SAGE End-to-End Results

The following table corresponds to the five test scenarios in Table 1 of the paper. `Base` is the Qwen3.5-397B-A17B baseline, while `SAGE` is the complete system using GPT-5.5. `Time` is the average per-task `execution_time_seconds`.

| Scenario      | Method | Result Success (%) | Joint Success (%) | Avg. Tokens / Trajectory |        Time |
| ------------- | ------ | -----------------: | ----------------: | -----------------------: | ----------: |
| retail6       | Base   |              26.53 |             22.45 |                  341,909 |     550.26s |
| retail6       | SAGE   |          **73.47** |         **73.47** |               **97,259** | **214.69s** |
| retail10      | Base   |              41.27 |             30.16 |                  236,814 | **208.19s** |
| retail10      | SAGE   |          **79.37** |         **61.90** |              **110,887** |     263.96s |
| order2        | Base   |              32.99 |             20.62 |                  324,969 |     497.21s |
| order2        | SAGE   |          **62.89** |         **54.64** |              **144,231** | **334.09s** |
| kitchen4      | Base   |              32.00 |             16.00 |                  480,564 |     609.01s |
| kitchen4      | SAGE   |          **76.00** |         **74.00** |              **113,180** | **268.25s** |
| restaurant5   | Base   |              28.00 |             22.00 |                  233,823 |     462.80s |
| restaurant5   | SAGE   |          **70.00** |         **68.00** |               **85,727** | **236.59s** |
| Weighted Avg. | Base   |              32.69 |             22.33 |                  320,110 |     459.22s |
| Weighted Avg. | SAGE   |          **71.20** |         **64.40** |              **115,493** | **274.43s** |

Bold indicates the better result within each scenario. Higher success rates are better, while lower token counts and times are better. The five scenarios contain 309 tasks in total. Averages are weighted by the number of tasks rather than computed as a macro-average over the five scenarios.

### Visual Evidence Ablation

The following table corresponds to Table 3 of the paper. The values are `retail1` Joint Success averaged over repeated runs:

| Visual Evidence          |   GPT-5.5 | Gemini-Flash | Gemini-Pro |
| ------------------------ | --------: | -----------: | ---------: |
| Keyframe only            |     37.50 |        46.88 |      50.00 |
| + Pair frame             |     68.75 |        59.38 |      50.00 |
| + Strict pair            |     59.38 |        65.62 |      53.12 |
| + Pair + evidence        |     65.62 |    **71.88** |      62.50 |
| + Pair + crop            |     68.75 |        62.50 |  **79.69** |
| + Pair + crop + evidence | **76.56** |        65.62 |      76.56 |

Simply changing the vision model does not reliably solve occlusion. Pairing evidence that anchors the target with unobstructed identity information is generally more reliable than using a single keyframe.

### Reporter Ablation

Table 4 of the paper removes the Reporter across the five test scenarios and compares Joint Success and average token usage:

| Scenario    | Joint with Reporter | Joint without Reporter | Improvement | Tokens with Reporter | Tokens without Reporter |
| ----------- | ------------------: | ---------------------: | ----------: | -------------------: | ----------------------: |
| retail6     |               73.47 |                  57.14 |      +16.33 |               97,259 |                 132,434 |
| retail10    |               61.90 |                  39.68 |      +22.22 |              110,887 |                 158,888 |
| order2      |               54.64 |                  48.45 |       +6.19 |              144,231 |                 177,654 |
| kitchen4    |               74.00 |                  52.00 |      +22.00 |              113,180 |                 167,749 |
| restaurant5 |               68.00 |                  48.00 |      +20.00 |               85,727 |                 132,268 |
| Overall     |           **64.40** |                  48.54 |  **+15.86** |          **115,493** |                 157,711 |

The Reporter provides both targeted repair signals and an evidence-compression interface: it improves overall Joint Success by 15.86 percentage points and reduces average token usage by 42,218 tokens per task.

### Planning Paradigms and Service Models

Table 5 of the paper compares three planning paradigms using the same GPT-5.5 model and `image_base64` visual interface:

| Planning Paradigm |      Tool |    Result |     Joint |   GT-call |  Avg task |
| ----------------- | --------: | --------: | --------: | --------: | --------: |
| SAGE              | **78.13** | **78.13** | **78.13** | **81.01** | **80.21** |
| ReAct             |     62.50 |     68.75 |     62.50 |     75.95 |     73.96 |
| Plan-and-Solve    |     62.50 |     62.50 |     62.50 |     69.62 |     65.63 |

Table 6 of the paper holds the complete SAGE pipeline fixed and changes only the service model for `retail1`:

| Service Model   |      Tool |    Result |     Joint |   GT-call |  Avg task |
| --------------- | --------: | --------: | --------: | --------: | --------: |
| DeepSeek-v4-pro |     71.88 |     71.88 |     71.88 |     77.22 |     73.96 |
| GPT-5.5         | **78.13** | **78.13** | **78.13** | **81.01** | **80.21** |

### Stage Latency in the Current Implementation

The following table is based on the 13 scenarios in the current implementation that have complete profiling, comprising 411 tasks in total. `TOTAL` uses `agent_response_time_seconds` and excludes the Simulated User and User Corrector. It therefore does not use the same timing definition as the paper's end-to-end `execution_time_seconds`.

| Stage                 | Avg. Time per Task | Share of Total Time |
| --------------------- | -----------------: | ------------------: |
| Frame Selector        |            27.438s |               9.16% |
| Box Locator           |            30.329s |              10.13% |
| Frames Reviewer       |            21.782s |               7.27% |
| Visual Recognition    |            12.209s |               4.08% |
| Supervisor            |            19.573s |               6.54% |
| Planner               |            69.175s |              23.10% |
| Executor              |            90.216s |              30.13% |
| Reporter              |            28.647s |               9.57% |
| Unattributed overhead |             0.078s |               0.02% |
| **TOTAL**             |       **299.447s** |         **100.00%** |

Together, the Executor and Planner account for 53.23% of total agent response time and are the main sources of latency in the current implementation. The Box Locator and Reviewer together account for 17.40%.

## Failure Cases

![Analysis of SAGE failure cases](https://huggingface.co/datasets/thomasrios/TYPERA/resolve/main/SAGE/SAGE_003.png)

The figure supplements the paper with four failure categories that were not elaborated in detail in the main text:

- **Visual Recognition Error**: During visual grounding, the target is identified as the wrong product. Although subsequent tool calls are structurally valid, the final state deviates from the ground truth.
- **User-Simulator Error**: The tool required by the ground truth has already been executed, but the simulated user continues down another branch, resulting in an extra state update.
- **GT not aligned with instruction**: The natural-language request asks the system to clear the entire order before adding dishes, but the ground truth retains the old order, creating a conflict between correctly following the instruction and matching the evaluation state.
- **Repair-Induced Error**: Definitions of terms such as “original price” are ambiguous, and the repair round adopts a different interpretation, causing the final choice to differ from the ground truth.

These cases show that end-to-end failures should not all be attributed to the service agent: visual anchoring, user simulation, annotation consistency, and repair semantics can all affect Joint Success.

## Reproducing the Experiments

The following instructions assume the current `egolink_track2_GML-MM-Group` working directory and primarily correspond to `run/multi_agent.py`, `config/*_config.py`, `run_all_scenarios.sh`, and `analysis_scripts/run_eval.sh`.

> [!CAUTION]
> Do not commit real API keys to the repository. All keys and service endpoints shown here are placeholders. Store real configuration values in `.env` at the repository root or in another private local environment.

The current project uses `image_base64` visual input exclusively: it first samples frames from the video at one-second intervals, then selects task-relevant frames, localizes and reviews visual targets, and finally encodes the images as base64 before sending them to the vision model. The legacy direct-video input and `final` keyframe modes are no longer available through the current run entry point.

### Project Structure

```text
.
├── config/                  # Configuration for User, Service, Visual, Tool, frame selection, bounding-box localization, and review
├── run/                     # Multi-agent pipeline, model adapters, visual pipeline, and performance statistics
├── tools/                   # Databases, tool definitions, and initialization logic for four scenario types
├── scenarios/
│   ├── final/               # All scenario tasks and ground truth
│   └── test_GT/             # Retained supplemental data; not read by the current run or evaluation entry points
├── videos/                  # Video files required for local reproduction
├── tests/                   # Regression tests for visual bounding-box localization, request parameters, and latency statistics
├── analysis_scripts/        # Result evaluation, statistical analysis, and plotting scripts
├── keyframe_test/           # Retained visual experiment code
├── SAGE/                    # Local backups of paper figures and failure cases
├── SAGE.pdf                 # Current paper draft
├── run_all_scenarios.sh     # Batch execution entry point
├── environment.yml          # Conda environment definition
├── requirements.txt         # Python dependencies
└── .env.example             # Template for private environment variables
```

The runtime-generated `results/`, `processed/`, and `eval_result/` directories are excluded by `.gitignore`.

## Quick Start

All commands are assumed to be run from the repository root.

```bash
conda env create -f environment.yml
conda activate egolink
python -m pip install -r requirements.txt
```

Basic environment checks:

```bash
python --version
ffmpeg -version
```

`environment.yml` uses Python 3.10 and installs FFmpeg, OpenCV, the OpenAI SDK, PyTorch, Transformers, Pillow, and the evaluation and plotting dependencies. If FFmpeg is unavailable, the code attempts to use OpenCV to obtain video durations and sample frames. For consistent results, however, using FFmpeg from the Conda environment is still recommended.

## API Configuration

The project's LLM and VLM calls use OpenAI-compatible endpoints. The current full pipeline primarily requires three configuration groups:

1. `SILICON_*`: Simulated User.
2. `VAPI_*`: User Corrector, Frame Selector, Box Locator, Frames Reviewer, Visual Agent, and Tool Agent.
3. `SERVICE_*`: Supervisor model under evaluation.

First, copy the environment-variable template:

```bash
cp .env.example .env
```

Then ensure that `.env` contains at least the following settings, replacing the placeholders with your private values:

```bash
# Simulated User
export SILICON_API_KEY="<YOUR_SILICON_API_KEY>"
export SILICON_API_URL="<YOUR_SILICON_OPENAI_COMPATIBLE_BASE_URL>"

# Visual and Tool Agents
export VAPI_API_KEY="<YOUR_VAPI_API_KEY>"
export VAPI_API_URL="<YOUR_VAPI_OPENAI_COMPATIBLE_BASE_URL>"

# Supervisor Under Evaluation
export SERVICE_MODEL_NAME="<YOUR_SERVICE_MODEL_NAME>"
export SERVICE_API_KEY="<YOUR_SERVICE_API_KEY>"
export SERVICE_API_BASE_URL="<YOUR_SERVICE_OPENAI_COMPATIBLE_BASE_URL>"

# Video Source
export VIDEO_MODE="local"
export VIDEO_LOCAL_PATH="./videos"
```

`run_all_scenarios.sh` automatically loads `.env` from the repository root. When invoking the Python entry point directly, first load the variables into the current shell:

```bash
set -a
source .env
set +a
```

It is recommended that `SERVICE_MODEL_NAME` in `.env` match the `--service_model_name` value passed on the command line. Some built-in model aliases select the corresponding provider adapter and its provider-specific environment variables. When reproducing experiments with a generic OpenAI-compatible service, use the `SERVICE_*` configuration above consistently.

| Module          | API Configuration                                            |
| --------------- | ------------------------------------------------------------ |
| Simulated User  | `SILICON_API_KEY`, `SILICON_API_URL`                         |
| User Corrector  | `VAPI_API_KEY`, `VAPI_API_URL`                               |
| Supervisor      | `SERVICE_API_KEY`, `SERVICE_API_BASE_URL`                    |
| Frame Selector  | Uses `VAPI_*` first, then falls back to `SKYCLAW_*` or the Service configuration |
| Visual Agent    | Uses `VAPI_*` first, then falls back to `SKYCLAW_*` or the Service configuration |
| Tool Agent      | `VAPI_API_KEY`, `VAPI_API_URL`                               |
| Frames Reviewer | `VAPI_API_KEY`, `VAPI_API_URL`                               |
| Box Locator     | `VISUAL_BOXED_API_KEY`, `VISUAL_BOXED_API_URL`; falls back to `VAPI_*` first if unset |

By default, the Box Locator tries the model list from the configuration file in order. To override the models and fallback order, set:

```bash
export VISUAL_BOXED_MODEL_NAMES="<PRIMARY_BOX_MODEL>,<FALLBACK_BOX_MODEL>"
```

Optional reasoning parameters include `FRAME_SELECTER_REASONING_EFFORT`, `VISUAL_AGENT_REASONING_EFFORT`, `VISUAL_BOXED_REASONING_EFFORT`, and `TOOL_PLANNER_REASONING_EFFORT`. The configuration is read when the Python process starts, so restart the Python process after modifying `.env`.

## Video Configuration

### Local Video Mode

By default, the repository's `videos/` directory is used:

```bash
export VIDEO_MODE="local"
export VIDEO_LOCAL_PATH="./videos"
```

All videos referenced by `scenarios/final/` are included in the current `videos/` directory. `VIDEO_LOCAL_PATH` is resolved relative to the directory from which the command is run, so starting the program from the repository root is recommended.

### URL Video Mode

URLs can also be used as video sources:

```bash
export VIDEO_MODE="url"
export VIDEO_URL_MAPPING='{
  "retail6.mp4": "https://example.com/videos/retail6.mp4",
  "retail10.mp4": "https://example.com/videos/retail10.mp4",
  "restaurant5.mp4": "https://example.com/videos/restaurant5.mp4"
}'
```

Mapping keys must exactly match the video filenames in the scenario JSON, including spaces and letter case. When running scenarios with multiple videos, the mapping must cover every filename they use. URL mode changes only the video source: the program still downloads or reads video frames and follows the `image_base64` pipeline rather than sending the full video directly to the vision model.

## Running a Single Scenario

The following example runs tasks 1 through 5 from `retail6`:

```bash
set -a
source .env
set +a

python run/multi_agent.py \
  --scenario retail \
  --scenario_number 6 \
  --service_model_name "<service_model_name>" \
  --multi_agent_user \
  --num_tasks 1,5
```

`--num_tasks` supports three forms:

| Value       | Meaning                                          |
| ----------- | ------------------------------------------------ |
| `0`         | Run all tasks in the scenario.                   |
| `N`         | Run the first N tasks in the scenario.           |
| `start,end` | Run tasks in the inclusive range `[start, end]`. |

The current CLI arguments are:

| Argument                     | Default Value        | Purpose                                                      |
| ---------------------------- | -------------------- | ------------------------------------------------------------ |
| `--service_model_name`       | `SERVICE_MODEL_NAME` | Specifies the model under evaluation and determines the result directory name. |
| `--scenario`                 | `retail`             | One of `retail`, `kitchen`, `restaurant`, or `order`.        |
| `--scenario_number`          | `1`                  | Scenario number.                                             |
| `--num_tasks`                | `0`                  | Selects all tasks, the first N tasks, or an inclusive range. |
| `--multi_agent_user`         | Off                  | Enables the User Corrector; enabled by default by the batch script. |
| `--test_visual`              | Off                  | Runs a visual-pipeline test that stops after the first visual recognition result. |
| `--tool_debug`               | Off                  | Retains internal Tool Agent and Visual Agent debugging fields in the result JSON. |
| `--stage_latency true/false` | `true`               | Controls whether per-stage latency and token statistics are recorded. |
| `--box true/false`           | `true`               | Controls whether target localization and review are enabled. |

To disable target localization for a controlled comparison:

```bash
python run/multi_agent.py \
  --scenario retail \
  --scenario_number 6 \
  --service_model_name "<service_model_name>" \
  --multi_agent_user \
  --num_tasks 1 \
  --box false
```

## Visual Module Smoke Test

Use `--test_visual` to check the frame sampling, frame selection, target localization, review, and visual recognition pipeline before running the full experiment:

```bash
python run/multi_agent.py \
  --scenario retail \
  --scenario_number 6 \
  --service_model_name "<service_model_name>" \
  --num_tasks 1,5 \
  --test_visual
```

This mode still runs the Simulated User and Supervisor. If routing first requires a tool condition to be resolved, it may temporarily enter the Tool Agent as well. Execution stops for a task after the first visual recognition result is produced, and the system saves a compact result. If visual recognition is never triggered within 10 rounds, the task is not written to the visual-test results.

Visual-test results are saved under:

```text
processed/visual_test/
└── {visual_model}/
    └── image_base64/
        └── {scenario}/
            └── {YYYYMMDD_HHMMSS}_{scenario}_easy.json
```

## Reproducing the Final Five Scenarios

| Type       | Scenarios             |
| ---------- | --------------------- |
| Retail     | `retail6`, `retail10` |
| Kitchen    | `kitchen4`            |
| Restaurant | `restaurant5`         |
| Order      | `order2`              |

The five scenarios contain 49, 63, 50, 50, and 97 tasks, respectively, for a total of 309 tasks.

Run all final evaluation scenarios:

```bash
bash run_all_scenarios.sh --final_eval
```

Quickly check the first task in each scenario:

```bash
bash run_all_scenarios.sh --final_eval --num_tasks 1
```

Without `--final_eval`, the script sequentially runs the repository's 10 Retail, 4 Kitchen, 5 Restaurant, and 2 Order scenarios: 21 scenarios and 1,037 tasks in total.

```bash
bash run_all_scenarios.sh
```

The batch script automatically loads `.env`, always uses the current `image_base64` entry point, and enables `--multi_agent_user`, per-stage latency statistics, and target localization by default.

## Output Files

### Complete Interaction Results

```text
results/
└── {service_model_name}/
    └── image_base64/
        └── {scenario}/
            └── {YYYYMMDD_HHMMSS}_{scenario}_easy.json
```

Each task result contains the `task_id`, conversation history, tool calls, number of rounds, token statistics, and execution time. Internal debugging fields are removed by default; use `--tool_debug` to retain them.

Timestamps in output filenames use UTC+8.

The program writes incrementally to the current output file, flushing each task to disk immediately after completion. If that same output file already exists, it is loaded and completed `task_id` values are skipped. Each CLI launch generates a new timestamped file, so rerunning the same command does not automatically resume the old file created by the previous run.

### Intermediate Visual Files

```text
processed/
├── frames/frame_1s/{video}/                         # Raw frames sampled at one-second intervals
├── frame_select/{scenario}/task{id}/request_{seq}/ # Selected frames and manifest
├── frame_select_boxed/{scenario}/task{id}/request_{seq}/
│                                                    # Boxed images, crops, fallback images, and manifest
└── visual_test/                                     # --test_visual output
```

Sampled frames, selected frames, and localization results are cached and reused. To regenerate part of the visual results from scratch, delete the corresponding `processed/` subdirectory after confirming that it is no longer needed.

### Stage Latency and Token Statistics

`--stage_latency` defaults to `true`. Its output directory is:

```text
processed/profiling/
└── {service_model_name}/
    └── image_base64/
        └── {scenario}/
            └── {YYYYMMDD_HHMMSS}_{scenario}_easy/
                ├── calls.jsonl
                ├── wall_events.jsonl
                ├── task_summaries.jsonl
                ├── stage_latency_summary.json
                └── stage_latency_summary.csv
```

The profiling leaf directory has the same name as the corresponding result JSON file without the `.json` extension. Statistics cover eight formal stages: Frame Selector, Box Locator, Frames Reviewer, Visual Agent, Supervisor, Tool Planner, Tool Executor, and Tool Reporter. The User Agent and User Corrector are recorded only in the raw events and are excluded from the formal stage summaries. Complete interactions use the `{service_model_name}` directory, while `--test_visual` uses the `{visual_model}` directory.

## Evaluation

The evaluation script recursively reads result files under `results/` that follow the timestamp naming format and evaluates tool calls and final database states against the `ground_truth` fields of the corresponding scenarios under `scenarios/final/`.

Evaluate a specified model directory:

```bash
bash analysis_scripts/run_eval.sh \
  --model_name "<service_model_name>"
```

Evaluate at most 10 tasks from each result file:

```bash
bash analysis_scripts/run_eval.sh \
  --model_name "<service_model_name>" \
  --num_samples 10
```

When `--model_name` is omitted, the script evaluates every directory under `results/` that contains new-format result files:

```bash
bash analysis_scripts/run_eval.sh
```

Evaluation results retain the same relative directory structure as `results/`:

```text
eval_result/
└── {service_model_name}/
    └── image_base64/
        └── {scenario}/
            ├── {YYYYMMDD_HHMMSS}_{scenario}_easy_eval.json
            └── summary.json
```

| Metric Field                                      | Meaning                                                      |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `tool_based.success_rate`                         | Whether a unique tool call with matching name and arguments can be found for every ground-truth tool call. |
| `result_based.success_rate`                       | Whether the database state after tool execution matches the reference result. |
| `joint_success.success_rate`                      | Whether both tool-based and result-based evaluation succeed. |
| `micro_tool_stats.micro_accuracy`                 | Micro recall across all ground-truth tool calls.             |
| `filtered_user_issue.filtered_joint_success_rate` | Joint success rate after excluding samples identified as simulated-user anomalies. |

## Regression Tests

Run the following command from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The current 23 tests cover:

- Frame Selector output parsing, prompts, and request parameters.
- Generic target localization, Reviewer acceptance/correction/rejection, and fallback to the original image.
- Model request parameters for the Visual Agent and Box Locator.
- Per-stage latency, nested-call latency, and token statistics.

## Pre-Reproduction Checklist

1. Create and activate the Python 3.10 environment using `environment.yml`.
2. Confirm that FFmpeg is available and the dependencies in `requirements.txt` are installed.
3. Create a private `.env` from `.env.example` and configure `SILICON_*`, `VAPI_*`, and `SERVICE_*`.
4. In local mode, verify that all required files are present under `videos/`. In URL mode, ensure that `VIDEO_URL_MAPPING` covers every video used by the scenarios being run.
5. Confirm that the visual input mode shown in the program logs is `image_base64`.
6. Complete a smoke test with `--num_tasks 1` or `--test_visual` first.
7. Run the final five scenarios with `bash run_all_scenarios.sh --final_eval`.
8. Generate evaluation results with `bash analysis_scripts/run_eval.sh --model_name <service_model_name>`.

