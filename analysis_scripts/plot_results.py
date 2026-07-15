# -*- coding: utf-8 -*-
"""
Plot evaluation result charts
1. Joint success rate by difficulty (LaTeX table)
2. Joint success rate by scenario (bar chart)
3. Error reason donut charts (by model and by scenario)
4. Average tokens per trajectory vs joint success rate
5. Average conversation rounds per trajectory vs joint success rate
6. Average tool calls per trajectory vs joint success rate
"""

import os
import re
import json
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from adjustText import adjust_text


# =========================
# Path Configuration
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

EVAL_RESULT_DIR = os.path.join(PROJECT_ROOT, "eval_result")
LOGO_DIR = os.path.join(CURRENT_DIR, "logo")
ERROR_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "error_analysis")
FIGURES_DIR = os.path.join(EVAL_RESULT_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================
# Global Plot Configuration
# =========================
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 18

# =========================
# Model Configuration
# =========================
MODEL_NAMES = {
    "glm-5v-turbo": "GLM-5V-Turbo",
    "qwen3-vl-225b": "Qwen3-VL-225B",
    "qwen3.6-plus": "Qwen3.6-Plus",
    "Qwen3.5-397B-A17B": "Qwen3.5-397B-A17B",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro",
    "kimi-k2.5": "Kimi-K2.5",
    "mimo-v2-omni": "MiMo-v2-omni",
    "doubao-seed-2-0-pro-260215": "Doubao-seed-2-0-pro"
}

LOGO_FILES = {
    "glm-5v-turbo": "chatglm-color.png",
    "qwen3-vl-225b": "qwen-color.png",
    "qwen3.6-plus": "qwen-color.png",
    "Qwen3.5-397B-A17B": "qwen-color.png",
    "gemini-3.1-pro-preview": "gemini-color.png",
    "kimi-k2.5": "kimi.png",
    "mimo-v2-omni": "xiaomimimo.png",
    "doubao-seed-2-0-pro-260215": "doubao-color.png"
}

MODEL_COLORS = {
    "glm-5v-turbo": "#E63946",
    "qwen3-vl-225b": "#457B9D",
    "qwen3.6-plus": "#2A9D8F",
    "Qwen3.5-397B-A17B": "#E9C46A",
    "gemini-3.1-pro-preview": "#9B59B6",
    "kimi-k2.5": "#3498DB",
    "mimo-v2-omni": "#E74C3C",
    "doubao-seed-2-0-pro-260215": "#1ABC9C"
}

LIGHT_COLORS = {
    "glm-5v-turbo": "#F5A5A8",
    "qwen3-vl-225b": "#8CBFD6",
    "qwen3.6-plus": "#72D4CC",
    "Qwen3.5-397B-A17B": "#F5DFA8",
    "gemini-3.1-pro-preview": "#C9A3D4",
    "kimi-k2.5": "#8DC6E8",
    "mimo-v2-omni": "#F5A593",
    "doubao-seed-2-0-pro-260215": "#6ED9CB"
}

MODEL_ORDER = list(MODEL_NAMES.keys())

SCENARIOS = ["retail", "restaurant", "order", "kitchen"]
DIFFICULTIES = ["easy", "hard", "static"]


# =========================
# Utility Functions
# =========================
def safe_read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read JSON: {path}, error={e}")
        return None


def list_model_dirs():
    """List model directories under eval_result that contain summary.json"""
    models = []
    if not os.path.exists(EVAL_RESULT_DIR):
        return models

    for name in os.listdir(EVAL_RESULT_DIR):
        model_dir = os.path.join(EVAL_RESULT_DIR, name)
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "summary.json")):
            models.append(name)

    # Keep only models defined in MODEL_NAMES, sorted by predefined order, extras at the end
    models = [m for m in models if m in MODEL_NAMES]
    ordered = [m for m in MODEL_ORDER if m in models]
    extras = [m for m in models if m not in ordered]
    return ordered + sorted(extras)


def get_model_summary(model_dir):
    path = os.path.join(EVAL_RESULT_DIR, model_dir, "summary.json")
    return safe_read_json(path)


def get_model_eval_files(model_dir):
    model_path = os.path.join(EVAL_RESULT_DIR, model_dir)
    if not os.path.exists(model_path):
        return []
    files = []
    for fn in os.listdir(model_path):
        if fn.endswith("_eval.json"):
            files.append(os.path.join(model_path, fn))
    return sorted(files)


def mean_or_zero(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else 0.0


def normalize_rate(x):
    """
    Some fields may be 0~1 or already 0~100.
    Normalize to percentage values.
    """
    if x is None:
        return 0.0
    try:
        x = float(x)
    except Exception:
        return 0.0

    if x <= 1.0:
        return x * 100
    return x


def load_logo_image(model, zoom=0.12):
    logo_file = LOGO_FILES.get(model)
    if not logo_file:
        return None
    logo_path = os.path.join(LOGO_DIR, logo_file)
    if not os.path.exists(logo_path):
        return None
    try:
        img = Image.open(logo_path).convert("RGBA")
        return OffsetImage(img, zoom=zoom)
    except Exception as e:
        print(f"[WARN] Failed to load logo: model={model}, error={e}")
        return None


# =========================
# Metric Extraction
# =========================
def get_four_rates_by_difficulty(summary):
    """
    Extract four metrics by difficulty: micro_accuracy, tool_based_success_rate, result_based_success_rate, joint_success_rate
    Weighted average using valid_scenarios to avoid bias from different scenario sample counts
    Returns: {difficulty: {"micro": x, "tool": x, "result": x, "joint": x}}
    """
    result = {d: {"micro": 0.0, "tool": 0.0, "result": 0.0, "joint": 0.0} for d in DIFFICULTIES}
    if not summary:
        return result

    all_results = summary.get("all_results", [])
    # Collect (rate, weight) pairs, weight is valid_scenarios
    buckets = {d: {"micro": [], "tool": [], "result": [], "joint": [], "weights": []} for d in DIFFICULTIES}

    for item in all_results:
        mode = item.get("mode")
        if mode in DIFFICULTIES:
            weight = item.get("valid_scenarios", 0)
            buckets[mode]["weights"].append(weight)
            buckets[mode]["micro"].append(normalize_rate(item.get("micro_accuracy", 0)))
            buckets[mode]["tool"].append(normalize_rate(item.get("tool_based_success_rate", 0)))
            buckets[mode]["result"].append(normalize_rate(item.get("result_based_success_rate", 0)))
            buckets[mode]["joint"].append(normalize_rate(item.get("joint_success_rate", 0)))

    for d in DIFFICULTIES:
        total_weight = sum(buckets[d]["weights"])
        if total_weight > 0:
            for key in ["micro", "tool", "result", "joint"]:
                result[d][key] = sum(r * w for r, w in zip(buckets[d][key], buckets[d]["weights"])) / total_weight
        else:
            for key in ["micro", "tool", "result", "joint"]:
                result[d][key] = mean_or_zero(buckets[d][key])
    return result


def get_joint_success_rate_by_difficulty(summary):
    result = {d: 0.0 for d in DIFFICULTIES}
    if not summary:
        return result

    rates = get_four_rates_by_difficulty(summary)
    for d in DIFFICULTIES:
        result[d] = rates[d]["joint"]
    return result


def get_joint_success_rate_by_scenario(summary):
    result = {s: 0.0 for s in SCENARIOS}
    if not summary:
        return result

    all_results = summary.get("all_results", [])
    # Collect (rate, weight) pairs, weight is valid_scenarios
    bucket = defaultdict(list)

    for item in all_results:
        scenario = item.get("scenario")
        if scenario in SCENARIOS:
            rate = normalize_rate(item.get("joint_success_rate", 0))
            weight = item.get("valid_scenarios", 0)
            bucket[scenario].append((rate, weight))

    for s in SCENARIOS:
        if bucket[s]:
            total_weight = sum(w for _, w in bucket[s])
            if total_weight > 0:
                result[s] = sum(r * w for r, w in bucket[s]) / total_weight
            else:
                result[s] = mean_or_zero([r for r, _ in bucket[s]])
    return result


def get_overall_joint_success_rate(summary):
    if not summary:
        return 0.0
    s = summary.get("summary", {})
    return normalize_rate(s.get("avg_joint_success_rate", 0))


def get_metrics_from_summary(summary):
    """
    Extract metrics from summary.json summary field
    Returns: avg_tokens, avg_input_tokens, avg_output_tokens, avg_rounds, avg_tool_calls
    """
    if not summary:
        return {"avg_tokens": 0.0, "avg_input_tokens": 0.0, "avg_output_tokens": 0.0, "avg_rounds": 0.0, "avg_tool_calls": 0.0}

    s = summary.get("summary", {})

    # Token consumption = input + output
    avg_input_tokens = s.get("avg_input_tokens", 0) or 0
    avg_output_tokens = s.get("avg_output_tokens", 0) or 0
    avg_tokens = avg_input_tokens + avg_output_tokens

    # Conversation rounds
    avg_rounds = s.get("avg_rounds_count", 0) or 0

    # Tool call count
    avg_tool_calls = s.get("avg_tool_calls_count", 0) or 0

    return {
        "avg_tokens": float(avg_tokens),
        "avg_input_tokens": float(avg_input_tokens),
        "avg_output_tokens": float(avg_output_tokens),
        "avg_rounds": float(avg_rounds),
        "avg_tool_calls": float(avg_tool_calls)
    }


# =========================
# Axis Model Logo Labels
# =========================
def add_model_logos_below_axis(ax, models, y_offset_axes=-0.16, zoom=0.065, fontsize=10):
    """
    Add logo + model name below x-axis
    Use coordinate transform for stable layout
    """
    xticks = ax.get_xticks()
    if len(xticks) < len(models):
        return

    for i, model in enumerate(models):
        x = xticks[i]

        # logo
        img = load_logo_image(model, zoom=zoom)
        if img is not None:
            ab = AnnotationBbox(
                img,
                (x, y_offset_axes + 0.035),
                xycoords=ax.get_xaxis_transform(),
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0
            )
            ax.add_artist(ab)

        # text
        ax.text(
            x, y_offset_axes - 0.02,
            MODEL_NAMES.get(model, model),
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=fontsize, fontweight="bold"
        )

    ax.set_xticklabels([])
    plt.subplots_adjust(bottom=0.30)


# =========================
# Chart 1: LaTeX Table by Difficulty
# =========================
def _find_best_and_second(values):
    """Return (best_idx, second_idx), best is max value index, second is second-max value index"""
    if not values:
        return None, None
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    best = sorted_indices[0] if len(sorted_indices) > 0 else None
    second = sorted_indices[1] if len(sorted_indices) > 1 else None
    return best, second


def _fmt_val(val, is_best, is_second):
    s = f"{val:.2f}"
    if is_best:
        return f"\\textbf{{{s}}}"
    if is_second:
        return f"\\underline{{{s}}}"
    return s


def plot_chart1_joint_success_by_difficulty(models):
    print("Generating Table 1: Four accuracy metrics by difficulty (LaTeX)")

    # Collect four metrics per model per difficulty
    model_data = {}
    for model in models:
        summary = get_model_summary(model)
        model_data[model] = get_four_rates_by_difficulty(summary)

    metrics = ["micro", "tool", "result", "joint"]
    metric_labels = ["Micro", "Tool", "Result", "Joint"]

    # Pre-compute best / second best per column
    col_best = {}   # (difficulty, metric) -> best_model_idx
    col_second = {}
    for d in DIFFICULTIES:
        for m in metrics:
            vals = [model_data[model][d][m] for model in models]
            best, second = _find_best_and_second(vals)
            col_best[(d, m)] = best
            col_second[(d, m)] = second

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"% \caption{Performance by Difficulty on EgoBench.}")
    lines.append(r"\label{tab:difficulty_results}")
    lines.append(r"\large")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{clcccccccccccc}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{4}{c}{\textbf{Easy}} & \multicolumn{4}{c}{\textbf{Hard}} & \multicolumn{4}{c}{\textbf{Static}} \\")
    lines.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}")
    lines.append(r" & \textbf{Model} & \textbf{Micro} & \textbf{Tool} & \textbf{Result} & \textbf{Joint} & \textbf{Micro} & \textbf{Tool} & \textbf{Result} & \textbf{Joint} & \textbf{Micro} & \textbf{Tool} & \textbf{Result} & \textbf{Joint} \\")
    lines.append(r"\midrule")

    for i, model in enumerate(models):
        logo_file = LOGO_FILES.get(model, "")
        logo_name = logo_file.replace(".png", "") if logo_file else ""
        display_name = MODEL_NAMES.get(model, model)

        row_vals = []
        for d in DIFFICULTIES:
            for j, m in enumerate(metrics):
                val = model_data[model][d][m]
                is_best = (col_best[(d, m)] == i)
                is_second = (col_second[(d, m)] == i)
                row_vals.append(_fmt_val(val, is_best, is_second))

        logo_part = f"\\includegraphics[height=0.3cm]{{logo/{logo_name}}}" if logo_name else ""
        line = f"{logo_part} & {display_name} & " + " & ".join(row_vals) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    latex_str = "\n".join(lines)

    save_path = os.path.join(FIGURES_DIR, "table1_difficulty_results.tex")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"LaTeX table saved to: {save_path}")


# =========================
# Chart 2: Bar Chart by Scenario
# =========================
def plot_chart2_joint_success_by_scenario(models):
    print("Plotting Chart 2: Joint success rate by scenario (bar chart)")

    x = np.arange(len(models))
    width = 0.18

    scenario_data = {s: [] for s in SCENARIOS}
    for model in models:
        summary = get_model_summary(model)
        rates = get_joint_success_rate_by_scenario(summary)
        for s in SCENARIOS:
            scenario_data[s].append(rates[s])

    fig, ax = plt.subplots(figsize=(15, 8))

    colors = {
        "retail": "#2E86AB",
        "restaurant": "#E55934",
        "order": "#3B1F2B",
        "kitchen": "#9BC53D"
    }

    for idx, scenario in enumerate(SCENARIOS):
        ax.bar(
            x + (idx - 1.5) * width,
            scenario_data[scenario],
            width=width,
            label=scenario,
            color=colors[scenario],
            edgecolor="white",
            linewidth=1.2
        )

    ax.set_xlabel("Models", fontsize=23)
    ax.set_ylabel("Joint Success Rate (%)", fontsize=23)
    # No title
    ax.set_xticks(x)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=18)

    ymax = max([max(v) if v else 0 for v in scenario_data.values()] + [5])
    ax.set_ylim(0, min(100, ymax + 10))

    add_model_logos_below_axis(ax, models, y_offset_axes=-0.14, zoom=0.06, fontsize=9)

    save_path = os.path.join(FIGURES_DIR, "chart2_joint_success_by_scenario.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Error Analysis Data Loading
# =========================
PIE_LABELS = ["Structural Non-Compliance", "Multimodal Perceptual Misinterpretations", "Hallucination", "Logical Fallacies", "Risky Operations", "Correct"]
PIE_KEYS = ["syntax", "multimodal", "hallucination", "logic", "over_action", "correct"]

# High-contrast color scheme; Correct: green
PIE_COLORS = ["#C62828", "#E64A19", "#FFA726", "#FDD835", "#7CB342", "#2E7D32"]


def _compute_pie_proportions(counts):
    """
    Calculate proportions (percentages) from error type counts, summing to 100%.
    Memory errors are merged into Logic.

    counts: {"total": N, "syntax_error": n, ...}
    Returns: {"syntax": %, "multimodal": %, "hallucination": %, "logic": %, "over_action": %, "correct": %}
    """
    total = counts.get("total", 0)
    if total == 0:
        return {}

    syntax = counts.get("syntax_error", 0)
    multimodal = counts.get("multimodal_error", 0)
    hallucination = counts.get("hallucination_error", 0)
    logic = counts.get("logic_error", 0) + counts.get("memory_error", 0)  # memory merged into logic
    over_op = counts.get("over_operation_error", 0)
    correct = counts.get("correct", 0)

    props = {}
    props["syntax"] = syntax / total * 100
    props["multimodal"] = multimodal / total * 100
    props["hallucination"] = hallucination / total * 100
    props["logic"] = logic / total * 100
    props["over_action"] = over_op / total * 100
    props["correct"] = correct / total * 100
    return props


def load_all_error_analysis():
    """
    Read detailed results from error_analysis/all_scenarios_error_analysis_details.json,
    aggregate error counts by scenario and model, then calculate pie chart proportions.

    Returns:
    (per_scenario, overall)
      per_scenario: {scenario: {model: {"syntax": %, ...}}}
      overall:      {model: {"syntax": %, ...}}
    """
    per_scenario = {s: defaultdict(dict) for s in SCENARIOS}
    overall_counts = {}

    json_path = os.path.join(ERROR_ANALYSIS_DIR, "all_scenarios_error_analysis_details.json")
    if not os.path.exists(json_path):
        print(f"[WARN] Error analysis JSON not found: {json_path}")
        return per_scenario, {}

    details = safe_read_json(json_path)
    if not details:
        return per_scenario, {}

    # Aggregate error type counts by (scenario_type, model)
    _EMPTY = lambda: {"total": 0, "syntax_error": 0, "multimodal_error": 0,
                      "hallucination_error": 0, "logic_error": 0, "memory_error": 0,
                      "over_operation_error": 0, "correct": 0}
    counts = defaultdict(lambda: defaultdict(_EMPTY))

    for entry in details:
        if "error" in entry:
            continue
        scenario_type = entry.get("scenario_type")
        model = entry.get("model")
        if not scenario_type or not model:
            continue

        for r in entry.get("results", []):
            error_type = r.get("error_type")
            if error_type in ("missing_gt", "missing_eval"):
                continue
            counts[scenario_type][model]["total"] += 1
            if error_type in counts[scenario_type][model]:
                counts[scenario_type][model][error_type] += 1

    # Calculate pie proportions per scenario per model
    for scenario_type in SCENARIOS:
        for model, cnt in counts[scenario_type].items():
            per_scenario[scenario_type][model] = _compute_pie_proportions(cnt)

    # Calculate pie proportions per model across scenarios
    all_models = set()
    for s in SCENARIOS:
        all_models.update(counts[s].keys())

    for model in all_models:
        merged = _EMPTY()
        for s in SCENARIOS:
            if model in counts[s]:
                for k in merged:
                    merged[k] += counts[s][model][k]
        overall_counts[model] = _compute_pie_proportions(merged)

    return per_scenario, overall_counts


# =========================
# Chart 3: Error Reason Donut Charts
# =========================
def _draw_single_donut(ax, props, model, logo_zoom=0.10, name_fontsize=15, pct_fontsize=11):
    """
    Draw a single donut chart on the given ax.
    props: {"syntax": %, "multimodal": %, ...}  summing to 100%
    """
    values = [props.get(k, 0) for k in PIE_KEYS]

    # Filter out zero-value items
    filtered_labels = []
    filtered_values = []
    filtered_colors = []
    for i, v in enumerate(values):
        if v > 0:
            filtered_labels.append(PIE_LABELS[i])
            filtered_values.append(v)
            filtered_colors.append(PIE_COLORS[i])

    if not filtered_values:
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                ha="center", va="center", fontsize=name_fontsize)
        ax.set_aspect("equal")
        ax.axis("off")
        return

    wedges, texts, autotexts = ax.pie(
        filtered_values,
        colors=filtered_colors,
        autopct=lambda pct: f"{pct:.1f}" if pct >= 4 else "",
        startangle=90,
        pctdistance=0.80,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=pct_fontsize, fontweight="bold"),
    )
    for t in autotexts:
        t.set_fontsize(pct_fontsize)
        t.set_color("#333333")

    # Center logo (offset upward)
    img = load_logo_image(model, zoom=logo_zoom)
    if img is not None:
        ab = AnnotationBbox(img, (0, 0.10), frameon=False, box_alignment=(0.5, 0.5), pad=0)
        ax.add_artist(ab)

    # Model name placed below center logo
    ax.text(0, -0.28, MODEL_NAMES.get(model, model),
            ha="center", va="center", fontsize=name_fontsize, fontweight="bold")


def plot_chart3a_pie_by_model(models):
    """Chart 3a: Overall error reason donut for 8 models, 2x4 grid"""
    print("Plotting Chart 3a: Overall error reason donut for 8 models (2x4)")

    _, overall_data = load_all_error_analysis()

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 11))

    for idx, model in enumerate(models):
        r, c = divmod(idx, ncols)
        if r >= nrows:
            break
        ax = axes[r][c]
        props = overall_data.get(model, {})
        _draw_single_donut(ax, props, model, logo_zoom=0.12, name_fontsize=20, pct_fontsize=17)

    # Hide extra subplots
    for idx in range(len(models), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    # Legend
    legend_elements = [plt.matplotlib.patches.Patch(
        facecolor=PIE_COLORS[i], edgecolor="white", label=PIE_LABELS[i]
    ) for i in range(len(PIE_LABELS))]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(PIE_LABELS),
               fontsize=20, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1.0])

    save_path = os.path.join(FIGURES_DIR, "chart3a_error_pie_by_model.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_chart3b_pie_by_model_scenario(models):
    """Chart 3b: Error reason donut for 4 scenarios x 8 models, 4x8 grid"""
    print("Plotting Chart 3b: Error reason donut for 4 scenarios x 8 models (4x8)")

    per_scenario, _ = load_all_error_analysis()

    scenario_labels = {
        "retail": "Retail", "restaurant": "Restaurant",
        "order": "Order", "kitchen": "Kitchen"
    }

    nrows = len(SCENARIOS)  # 4
    ncols = len(models)     # 8
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))

    for row, scenario in enumerate(SCENARIOS):
        for col, model in enumerate(models):
            ax = axes[row][col] if nrows > 1 else axes[col]
            props = per_scenario[scenario].get(model, {})
            _draw_single_donut(ax, props, model,
                               logo_zoom=0.07, name_fontsize=9, pct_fontsize=12)
            # Add scenario label to first column
            if col == 0:
                ax.text(-0.25, 0.5, scenario_labels.get(scenario, scenario),
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=21, fontweight="bold", rotation=90)

    # Legend
    legend_elements = [plt.matplotlib.patches.Patch(
        facecolor=PIE_COLORS[i], edgecolor="white", label=PIE_LABELS[i]
    ) for i in range(len(PIE_LABELS))]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(PIE_LABELS),
               fontsize=18, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0.03, 0.03, 1, 1.0])

    save_path = os.path.join(FIGURES_DIR, "chart3b_error_pie_by_model_scenario.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_chart3_error_pie(models):
    """Plot all error reason donut charts"""
    plot_chart3a_pie_by_model(models)
    plot_chart3b_pie_by_model_scenario(models)


# =========================
# Scatter Plot Utility Functions
# =========================
def annotate_scatter_with_logo(ax, x, y, model, text_offset=(0, -12), zoom=0.11, fontsize=10):
    img = load_logo_image(model, zoom=zoom)
    if img is not None:
        ab = AnnotationBbox(
            img, (x, y),
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0
        )
        ax.add_artist(ab)

    ax.annotate(
        MODEL_NAMES.get(model, model),
        (x, y),
        textcoords="offset points",
        xytext=text_offset,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold"
    )


def annotate_scatter_points(ax, models, xs, ys, fontsize=16):
    """
    Use adjustText library to auto-annotate scatter points with model names
    - Text color matches point color
    - Auto-adjust position to avoid text-text and text-point overlaps, staying within plot bounds
    """
    texts = []

    for model, x, y in zip(models, xs, ys):
        color = MODEL_COLORS.get(model, "#000000")
        model_name = MODEL_NAMES.get(model, model)

        # Add text annotation, initial position at the point
        text = ax.text(
            x, y,
            model_name,
            fontsize=fontsize,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            zorder=10
        )
        texts.append(text)

    # Use adjustText to auto-adjust text positions
    adjust_text(
        texts,
        x=xs,
        y=ys,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0, alpha=0),
        expand_points=(10.0, 10.0),
        force_points=(5.0, 5.0),
        force_text=(4.0, 4.0),
        ensure_inside_view=False,
    )


def get_axis_padding(vals, ratio=0.08):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0, 1
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmin, vmax):
        pad = vmax * ratio if vmax != 0 else 1
    else:
        pad = (vmax - vmin) * ratio
    return vmin - pad, vmax + pad


# =========================
# Chart 4: Tokens vs Success
# =========================
def plot_chart4_tokens_vs_success(models):
    print("Plotting Chart 4: Avg token consumption vs joint success rate")

    xs, ys = [], []

    for model in models:
        summary = get_model_summary(model)
        metrics = get_metrics_from_summary(summary)
        xs.append(metrics["avg_tokens"])
        ys.append(get_overall_joint_success_rate(summary))

    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw with large points
    for model, x, y in zip(models, xs, ys):
        color = MODEL_COLORS.get(model, "#000000")
        ax.scatter(x, y, c=color, s=150, edgecolors="white", linewidths=1.5, zorder=5)

    ax.set_xlabel("Average Tokens per Trajectory (Input + Output)", fontsize=28)
    ax.set_ylabel("Joint Success Rate (%)", fontsize=28)
    ax.tick_params(labelsize=22)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set x-axis to k units
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(round(x/1000))}k"))

    xmin, xmax = get_axis_padding(xs)
    ymin, ymax = get_axis_padding(ys)
    ax.set_xlim(max(0, xmin), xmax)
    ax.set_ylim(max(0, ymin - 2), min(100, ymax + 4))

    save_path = os.path.join(FIGURES_DIR, "chart4_tokens_vs_success.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Chart 4a: Input Tokens vs Success
# =========================
def plot_chart4a_input_tokens_vs_success(models):
    print("Plotting Chart 4a: Avg input token consumption vs joint success rate")

    xs, ys = [], []

    for model in models:
        summary = get_model_summary(model)
        metrics = get_metrics_from_summary(summary)
        xs.append(metrics["avg_input_tokens"])
        ys.append(get_overall_joint_success_rate(summary))

    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw with large points
    for model, x, y in zip(models, xs, ys):
        color = MODEL_COLORS.get(model, "#000000")
        ax.scatter(x, y, c=color, s=150, edgecolors="white", linewidths=1.5, zorder=5)

    ax.set_xlabel("Average Input Tokens per Trajectory", fontsize=28)
    ax.set_ylabel("Joint Success Rate (%)", fontsize=28)
    ax.tick_params(labelsize=22)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set x-axis to k units
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(round(x/1000))}k"))

    xmin, xmax = get_axis_padding(xs)
    ymin, ymax = get_axis_padding(ys)
    ax.set_xlim(max(0, xmin), xmax)
    ax.set_ylim(max(0, ymin - 2), min(100, ymax + 4))

    save_path = os.path.join(FIGURES_DIR, "chart4a_input_tokens_vs_success.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Chart 4b: Output Tokens vs Success
# =========================
def plot_chart4b_output_tokens_vs_success(models):
    print("Plotting Chart 4b: Avg output token consumption vs joint success rate")

    xs, ys = [], []

    for model in models:
        summary = get_model_summary(model)
        metrics = get_metrics_from_summary(summary)
        xs.append(metrics["avg_output_tokens"])
        ys.append(get_overall_joint_success_rate(summary))

    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw with large points
    for model, x, y in zip(models, xs, ys):
        color = MODEL_COLORS.get(model, "#000000")
        ax.scatter(x, y, c=color, s=150, edgecolors="white", linewidths=1.5, zorder=5)

    ax.set_xlabel("Average Output Tokens per Trajectory", fontsize=28)
    ax.set_ylabel("Joint Success Rate (%)", fontsize=28)
    ax.tick_params(labelsize=22)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set x-axis to k units
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(round(x/1000))}k"))

    xmin, xmax = get_axis_padding(xs)
    ymin, ymax = get_axis_padding(ys)
    ax.set_xlim(max(0, xmin), xmax)
    ax.set_ylim(max(0, ymin - 2), min(100, ymax + 4))

    save_path = os.path.join(FIGURES_DIR, "chart4b_output_tokens_vs_success.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Chart 5: Rounds vs Success
# =========================
def plot_chart5_rounds_vs_success(models):
    print("Plotting Chart 5: Avg conversation rounds vs joint success rate")

    xs, ys = [], []

    for model in models:
        summary = get_model_summary(model)
        metrics = get_metrics_from_summary(summary)
        xs.append(metrics["avg_rounds"])
        ys.append(get_overall_joint_success_rate(summary))

    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw with large points
    for model, x, y in zip(models, xs, ys):
        color = MODEL_COLORS.get(model, "#000000")
        ax.scatter(x, y, c=color, s=150, edgecolors="white", linewidths=1.5, zorder=5)

    ax.set_xlabel("Average User-Agent Conversation Rounds per Trajectory", fontsize=28)
    ax.set_ylabel("Joint Success Rate (%)", fontsize=28)
    ax.tick_params(labelsize=22)
    ax.grid(True, linestyle="--", alpha=0.3)

    xmin, xmax = get_axis_padding(xs)
    ymin, ymax = get_axis_padding(ys)
    ax.set_xlim(max(0, xmin), xmax)
    ax.set_ylim(max(0, ymin - 2), min(100, ymax + 4))

    save_path = os.path.join(FIGURES_DIR, "chart5_rounds_vs_success.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Chart 6: Tool Calls vs Success
# =========================
def plot_chart6_tool_calls_vs_success(models):
    print("Plotting Chart 6: Avg tool calls vs joint success rate")

    xs, ys = [], []

    for model in models:
        summary = get_model_summary(model)
        metrics = get_metrics_from_summary(summary)
        xs.append(metrics["avg_tool_calls"])
        ys.append(get_overall_joint_success_rate(summary))

    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw with large points
    for model, x, y in zip(models, xs, ys):
        color = MODEL_COLORS.get(model, "#000000")
        ax.scatter(x, y, c=color, s=150, edgecolors="white", linewidths=1.5, zorder=5)

    ax.set_xlabel("Average Tool Calls per Trajectory", fontsize=28)
    ax.set_ylabel("Joint Success Rate (%)", fontsize=28)
    ax.tick_params(labelsize=22)
    ax.grid(True, linestyle="--", alpha=0.3)

    xmin, xmax = get_axis_padding(xs)
    ymin, ymax = get_axis_padding(ys)
    ax.set_xlim(max(0, xmin), xmax)
    ax.set_ylim(max(0, ymin - 2), min(100, ymax + 4))

    save_path = os.path.join(FIGURES_DIR, "chart6_tool_calls_vs_success.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Main Function
# =========================
def main():
    print("=" * 60)
    print("Starting evaluation result chart generation")
    print(f"EVAL_RESULT_DIR   : {EVAL_RESULT_DIR}")
    print(f"LOGO_DIR          : {LOGO_DIR}")
    print(f"ERROR_ANALYSIS_DIR: {ERROR_ANALYSIS_DIR}")
    print(f"FIGURES_DIR       : {FIGURES_DIR}")
    print("=" * 60)

    models = list_model_dirs()
    if not models:
        print("[ERROR] No valid model directories found under eval_result (containing summary.json)")
        return

    print("Detected models:")
    for m in models:
        print(f" - {m}")

    plot_chart1_joint_success_by_difficulty(models)
    plot_chart2_joint_success_by_scenario(models)
    plot_chart3_error_pie(models)
    plot_chart4_tokens_vs_success(models)
    plot_chart4a_input_tokens_vs_success(models)
    plot_chart4b_output_tokens_vs_success(models)
    plot_chart5_rounds_vs_success(models)
    plot_chart6_tool_calls_vs_success(models)

    print("=" * 60)
    print("All charts generated.")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
