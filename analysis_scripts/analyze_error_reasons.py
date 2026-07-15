#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-Scenario Error Reason Analysis Script - Based on eval_result data

Supported scenarios: retail, restaurant, order, kitchen

Error reasons are determined for each sample in the following order:
1. Syntax format error
2. Multimodal recognition error (whether query tool calls match key/value)
3. Hallucination issue (incorrect user_id parameter)
4. Tool call logic reasoning error (matches < total_gt_calls, difference >= 2)
5. Memory/common sense error (matches < total_gt_calls, difference == 1)
6. Over-operation error (matches == total_gt_calls but result_based failed)

Correct sample: both tool-based evaluation and result-based evaluation are correct
"""

import json
import os
import sys
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.retail.retail_db import RetailDB
from tools.retail.retail_init import (
    retail_init_data1, retail_init_data2, retail_init_data3, retail_init_data4,
    retail_init_data5, retail_init_data6, retail_init_data7, retail_init_data8,
    retail_init_data9, retail_init_data10
)
from tools.restaurant.restaurant_db import RestaurantDB
from tools.restaurant.restaurant_init import restaurant_init_data, restaurant_init_data5
from tools.order.order_db import OrderDB
from tools.order.order_init import order_init_data
from tools.kitchen.kitchen_db import KitchenDB
from tools.kitchen.kitchen_init import kitchen_init_data


MODELS = [
    'qwen3-vl-225b', 'Qwen3.5-397B-A17B', 'gemini-3.1-pro-preview',
    'kimi-k2.5', 'mimo-v2-omni', 'qwen3.6-plus', 'glm-5v-turbo', 'doubao-seed-2-0-pro-260215'
]

ERROR_TYPES = {
    'syntax_error': 'Syntax format error in tool calls',
    'multimodal_error': 'Multimodal recognition error',
    'hallucination_error': 'Hallucination issue (incorrect user_id parameter)',
    'logic_error': 'Tool call logic reasoning error',
    'memory_error': 'Memory or common sense error',
    'over_operation_error': 'Over-operation error',
    'correct': 'Correct'
}

MODES = ['easy', 'hard', 'static']

SCENARIO_CONFIG = {
    'retail': {
        'scenario_numbers': list(range(1, 11)),
        'init_data_map': {
            1: retail_init_data1, 2: retail_init_data2, 3: retail_init_data3,
            4: retail_init_data4, 5: retail_init_data5, 6: retail_init_data6,
            7: retail_init_data7, 8: retail_init_data8, 9: retail_init_data9,
            10: retail_init_data10
        },
        'db_class': RetailDB,
        'query_tools': {
            'get_price', 'get_tax_rate', 'get_category', 'get_discount', 'get_nutrition',
            'find_products_by_nutritional_characteristic', 'find_products_by_taste',
            'find_products_by_country_of_origin', 'find_products_by_price_range',
            'list_discounted_products'
        },
    },
    'restaurant': {
        'scenario_numbers': list(range(1, 6)),
        'init_data_map': {
            1: restaurant_init_data, 2: restaurant_init_data,
            3: restaurant_init_data, 4: restaurant_init_data,
            5: restaurant_init_data5
        },
        'db_class': RestaurantDB,
        'query_tools': {
            'get_dish_price', 'get_dish_discount', 'get_dish_nutrition',
            'get_dish_allergens', 'get_tax_rate', 'get_dish_taste_profile',
            'get_user_order_summary', 'get_set_meal_details',
            'find_dishes_by_category', 'find_dishes_by_nutritional_tag',
            'find_dishes_by_taste', 'find_set_meals_containing_dish',
            'list_all_discounted_dishes', 'filter_dishes_by_price_range',
            'compute_total_payment', 'compute_total_tax', 'compute_total_nutrition'
        },
    },
    'order': {
        'scenario_numbers': list(range(1, 3)),
        'init_data_map': {
            1: order_init_data, 2: order_init_data
        },
        'db_class': OrderDB,
        'query_tools': {
            'get_dish_price', 'get_dish_discount', 'get_dish_nutrition',
            'get_dish_allergens', 'get_tax_rate', 'get_dish_taste_profile',
            'get_user_order_summary', 'get_set_meal_details',
            'find_dishes_by_category', 'find_dishes_by_nutritional_tag',
            'find_dishes_by_taste', 'find_set_meals_containing_dish',
            'list_all_discounted_dishes', 'filter_dishes_by_price_range',
            'compute_total_payment', 'compute_total_tax', 'compute_total_nutrition'
        },
    },
    'kitchen': {
        'scenario_numbers': list(range(1, 5)),
        'init_data_map': {
            1: kitchen_init_data, 2: kitchen_init_data,
            3: kitchen_init_data, 4: kitchen_init_data
        },
        'db_class': KitchenDB,
        'query_tools': {
            'get_current_menu', 'get_cooking_steps', 'get_recipe_allergens',
            'get_recipe_taste', 'get_recipe_ingredients', 'get_ingredient_shelf_life',
            'get_ingredient_location', 'get_ingredient_nutrition', 'get_ingredient_quantity',
            'get_current_shopping_list', 'get_recipe_nutritional_characteristics',
            'find_recipes_by_allergen', 'find_recipes_by_taste',
            'find_recipes_by_ingredient', 'find_ingredients_by_expiry_date',
            'find_ingredients_by_location', 'find_recipes_by_nutritional_characteristics',
            'get_ingredients_by_category', 'find_ingredient_category',
            'get_all_recipe_names', 'get_all_ingredient_names',
            'tally_total_nutritional_characteristics', 'tally_total_tastes',
            'compute_total_nutritions'
        },
    },
}

ALL_SCENARIO_TYPES = list(SCENARIO_CONFIG.keys())


def load_init_db(scenario_type: str, scenario_number: int):
    """Initialize database based on scenario type and number"""
    config = SCENARIO_CONFIG[scenario_type]
    db = config['db_class']()
    init_data = config['init_data_map'].get(scenario_number)
    if init_data:
        db.init_from_json(init_data)
    return db


def load_ground_truth(scenario_type: str, scenario_number: int) -> List[Dict]:
    """Load ground truth file"""
    gt_path = os.path.join(
        os.path.dirname(__file__), '..', 'scenarios', 'final',
        f'{scenario_type}{scenario_number}.json'
    )
    gt_path = os.path.normpath(gt_path)

    if not os.path.exists(gt_path):
        return []

    with open(gt_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_interaction_log(model_name: str, scenario_type: str, scenario_number: int, mode: str) -> Optional[List[Dict]]:
    """Load interaction log"""
    log_path = os.path.join(
        os.path.dirname(__file__), '..', 'results', model_name,
        f'{scenario_type}{scenario_number}_{mode}.json'
    )
    log_path = os.path.normpath(log_path)

    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_eval_result(model_name: str, scenario_type: str, scenario_number: int, mode: str) -> Optional[Dict]:
    """Load evaluation result"""
    eval_path = os.path.join(
        os.path.dirname(__file__), '..', 'eval_result', model_name,
        f'{scenario_type}{scenario_number}_{mode}_eval.json'
    )
    eval_path = os.path.normpath(eval_path)

    if not os.path.exists(eval_path):
        return None

    with open(eval_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_all_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
    """Extract all tool calls (flatten)"""
    all_calls = []

    for entry in tool_calls:
        if not isinstance(entry, dict):
            continue

        if "calls" in entry and isinstance(entry["calls"], list):
            for call in entry["calls"]:
                if isinstance(call, dict):
                    all_calls.append({
                        "tool_name": call.get("tool_name") or call.get("name"),
                        "parameters": call.get("parameters", {})
                    })
        elif "call" in entry:
            call = entry.get("call", {})
            if isinstance(call, dict):
                all_calls.append({
                    "tool_name": call.get("tool_name") or call.get("name"),
                    "parameters": call.get("parameters", {})
                })

    return all_calls


def check_syntax_error(tool_calls: List[Dict]) -> Tuple[bool, str]:
    """
    1. Check if tool call syntax format is incorrect

    Returns: (has_syntax_error, error_description)
    """
    if not tool_calls:
        return True, "No tool calls"

    for i, call_entry in enumerate(tool_calls):
        if not isinstance(call_entry, dict):
            return True, f"Tool call entry {i + 1} has incorrect format"

        if "calls" in call_entry:
            calls = call_entry.get("calls", [])
            if not isinstance(calls, list):
                return True, f"Tool {i + 1} 'calls' field is not a list"

            for j, call in enumerate(calls):
                if not isinstance(call, dict):
                    return True, f"Tool {i + 1} call {j + 1} has incorrect format"
                if "tool_name" not in call and "name" not in call:
                    return True, f"Tool {i + 1} call {j + 1} missing tool_name"
                if "parameters" not in call:
                    return True, f"Tool {i + 1} call {j + 1} missing parameters"
        elif "call" in call_entry:
            call = call_entry.get("call", {})
            if not isinstance(call, dict):
                return True, f"Tool {i + 1} 'call' field has incorrect format"
            if "tool_name" not in call and "name" not in call:
                return True, f"Tool {i + 1} missing tool_name"
            if "parameters" not in call:
                return True, f"Tool {i + 1} missing parameters"
        else:
            return True, f"Tool call entry {i + 1} has incorrect format"

    return False, ""


def check_multimodal_error(
    gt_entry: Dict,
    interaction_calls: List[Dict],
    query_tools: set,
    scenario_type: str = None,
    scenario_number: int = None
) -> Tuple[bool, str]:
    """
    2. Check multimodal recognition error

    By comparing whether all query tool calls match the multimodal information
    in the ground truth file, i.e., whether the tested agent's tool call parameters
    contain the key and value from the scenario file. There must be tested tool calls,
    and if the value list has multiple values, tool calls containing all these
    parameters must be present.

    For order scenarios, also check if the restaurant selection is correct:
    - order1 should select "Annie Italian Restaurant"
    - order2 should select "Mediterranean Greek Restaurant"

    Returns: (has_multimodal_error, error_description)
    """
    # For order scenarios, first check if restaurant selection is correct
    if scenario_type == "order" and scenario_number in [1, 2]:
        expected_restaurant = (
            "Annie Italian Restaurant" if scenario_number == 1
            else "Mediterranean Greek Restaurant"
        )

        # Find the last tool call with restaurant_name parameter
        last_restaurant_choice = None
        for call in reversed(interaction_calls):
            params = call.get("parameters", {})
            if "restaurant_name" in params and isinstance(params["restaurant_name"], str):
                last_restaurant_choice = params["restaurant_name"]
                break

        if last_restaurant_choice is not None and last_restaurant_choice != expected_restaurant:
            return True, f"Incorrect restaurant selection: expected '{expected_restaurant}', got '{last_restaurant_choice}'"

    key = gt_entry.get("key")
    value = gt_entry.get("value", [])

    if not key or not value:
        return False, "No multimodal recognition requirement"

    if not interaction_calls:
        return False, "No tool calls"

    matched_values = set()

    for call in interaction_calls:
        tool_name = call.get("tool_name", "")
        if tool_name not in query_tools:
            continue

        params = call.get("parameters", {})
        if key not in params:
            continue

        param_value = params[key]
        for expected_value in value:
            if isinstance(param_value, str) and isinstance(expected_value, str):
                if param_value.lower().strip() == expected_value.lower().strip():
                    matched_values.add(expected_value)
                elif expected_value.lower() in param_value.lower():
                    matched_values.add(expected_value)
                elif fuzzy_match_name(param_value, expected_value):
                    matched_values.add(expected_value)
            elif param_value == expected_value:
                matched_values.add(expected_value)

    unmatched_values = set(value) - matched_values
    if unmatched_values:
        return True, f"Failed to recognize {key}={list(unmatched_values)}"

    return False, ""


def fuzzy_match_name(param_value: str, expected_value: str) -> bool:
    """Fuzzy match name"""
    param_lower = param_value.lower().strip()
    expected_lower = expected_value.lower().strip()

    if param_lower in expected_lower or expected_lower in param_lower:
        return True

    param_words = set(param_lower.split())
    expected_words = set(expected_lower.split())
    return bool(param_words & expected_words)


def check_hallucination_error(gt_entry: Dict, interaction_calls: List[Dict]) -> Tuple[bool, str]:
    """
    3. Check hallucination error (incorrect user_id parameter)

    Returns: (has_hallucination_error, error_description)
    """
    instruction = gt_entry.get("Instruction", "")
    user_id_match = re.search(r'User ID:\s*(\w+)', instruction)

    if not user_id_match:
        return False, "No user_id requirement"

    correct_user_id = user_id_match.group(1)

    wrong_user_ids = []
    for call in interaction_calls:
        params = call.get("parameters", {})
        if "user_id" in params and params["user_id"] != correct_user_id:
            wrong_user_ids.append(params["user_id"])

    if wrong_user_ids:
        return True, f"user_id should be '{correct_user_id}', but found {set(wrong_user_ids)}"

    return False, ""


def analyze_scenario(
    scenario_type: str,
    gt_entry: Dict,
    interaction_entry: Dict,
    eval_detail: Dict,
    scenario_number: int
) -> Dict:
    """
    Analyze error reasons for a single scenario

    First determine if successful (tool_based_success AND result_based_success),
    if successful, classify as correct; if failed, determine error reasons in order:
    1. Syntax error -> 2. Multimodal error -> 3. Hallucination error -> 4. Logic error -> 5. Memory error -> 6. Over-operation
    """
    result = {
        "scenario_type": scenario_type,
        "scenario_number": scenario_number,
        "task_id": gt_entry.get("task_id", scenario_number),
        "error_type": None,
        "error_desc": None,
        "details": {},
        "step_checks": {},
        "eval_data": {
            "matches": eval_detail.get("tool_based", {}).get("matches", 0),
            "total_gt_calls": eval_detail.get("tool_based", {}).get("total_gt_calls", 0),
            "tool_based_success": eval_detail.get("tool_based", {}).get("success", False),
            "result_based_success": eval_detail.get("result_based", {}).get("success", False)
        }
    }

    interaction_calls_raw = interaction_entry.get("tool_calls", [])
    interaction_calls = extract_all_tool_calls(interaction_calls_raw)

    matches = eval_detail.get("tool_based", {}).get("matches", 0)
    total_gt_calls = eval_detail.get("tool_based", {}).get("total_gt_calls", 0)
    tool_based_success = eval_detail.get("tool_based", {}).get("success", False)
    result_based_success = eval_detail.get("result_based", {}).get("success", False)

    # First check if successful: correct if both tool_based and result_based succeed
    if tool_based_success and result_based_success:
        result["error_type"] = "correct"
        result["error_desc"] = "No error"
        result["details"]["correct"] = True
        # Still record check results as additional information
        has_syntax_error, syntax_desc = check_syntax_error(interaction_calls_raw)
        result["step_checks"]["syntax_error"] = has_syntax_error
        query_tools = SCENARIO_CONFIG[scenario_type]['query_tools']
        has_multimodal_error, multimodal_desc = check_multimodal_error(
            gt_entry, interaction_calls, query_tools, scenario_type, scenario_number
        )
        result["step_checks"]["multimodal_error"] = has_multimodal_error
        has_hallucination_error, hallucination_desc = check_hallucination_error(gt_entry, interaction_calls)
        result["step_checks"]["hallucination_error"] = has_hallucination_error
        if has_multimodal_error:
            result["details"]["multimodal_note"] = multimodal_desc
        if has_hallucination_error:
            result["details"]["hallucination_note"] = hallucination_desc
        result["step_checks"]["logic_error"] = False
        result["step_checks"]["memory_error"] = False
        result["step_checks"]["over_operation_error"] = False
        return result

    # Below: error reason determination for failed samples
    has_syntax_error, syntax_desc = check_syntax_error(interaction_calls_raw)
    result["step_checks"]["syntax_error"] = has_syntax_error
    if has_syntax_error:
        result["error_type"] = "syntax_error"
        result["error_desc"] = syntax_desc
        result["details"]["syntax_error"] = syntax_desc
        return result

    query_tools = SCENARIO_CONFIG[scenario_type]['query_tools']
    has_multimodal_error, multimodal_desc = check_multimodal_error(
        gt_entry, interaction_calls, query_tools, scenario_type, scenario_number
    )
    result["step_checks"]["multimodal_error"] = has_multimodal_error
    if has_multimodal_error:
        result["error_type"] = "multimodal_error"
        result["error_desc"] = multimodal_desc
        result["details"]["multimodal_error"] = multimodal_desc
        return result

    has_hallucination_error, hallucination_desc = check_hallucination_error(gt_entry, interaction_calls)
    result["step_checks"]["hallucination_error"] = has_hallucination_error
    if has_hallucination_error:
        result["error_type"] = "hallucination_error"
        result["error_desc"] = hallucination_desc
        result["details"]["hallucination_error"] = hallucination_desc
        return result

    diff = total_gt_calls - matches

    if diff >= 2:
        result["error_type"] = "logic_error"
        result["error_desc"] = f"Insufficient tool call matches: matches={matches}, total_gt_calls={total_gt_calls}, missing {diff}"
        result["details"]["logic_error"] = {
            "matches": matches,
            "total_gt_calls": total_gt_calls,
            "missing": diff
        }
        result["step_checks"]["logic_error"] = True
        result["step_checks"]["memory_error"] = False
        result["step_checks"]["over_operation_error"] = False
        return result

    if diff == 1:
        result["error_type"] = "memory_error"
        result["error_desc"] = f"Only the last tool call missed: matches={matches}, total_gt_calls={total_gt_calls}"
        result["details"]["memory_error"] = {
            "matches": matches,
            "total_gt_calls": total_gt_calls,
            "missing": 1
        }
        result["step_checks"]["logic_error"] = False
        result["step_checks"]["memory_error"] = True
        result["step_checks"]["over_operation_error"] = False
        return result

    if matches == total_gt_calls and not result_based_success:
        result["error_type"] = "over_operation_error"
        result["error_desc"] = "Tool calls correct but result incorrect"
        result["details"]["over_operation_error"] = {
            "matches": matches,
            "total_gt_calls": total_gt_calls,
            "tool_based_success": tool_based_success,
            "result_based_success": result_based_success
        }
        result["step_checks"]["logic_error"] = False
        result["step_checks"]["memory_error"] = False
        result["step_checks"]["over_operation_error"] = True
        return result

    result["error_type"] = "unknown"
    result["error_desc"] = (
        f"Unknown case: matches={matches}, total_gt_calls={total_gt_calls}, "
        f"tool_success={tool_based_success}, result_success={result_based_success}"
    )

    return result


def analyze_model_performance(
    model_name: str,
    scenario_type: str,
    scenario_number: int,
    mode: str,
    ground_truth: List[Dict],
    eval_result: Dict
) -> Dict:
    """Analyze model performance in a specific scenario"""
    interaction_log = load_interaction_log(model_name, scenario_type, scenario_number, mode)

    if interaction_log is None:
        return {"error": f"Result file not found: {scenario_type}{scenario_number}_{mode}.json"}

    results = []

    for i, interaction_entry in enumerate(interaction_log):
        task_id = interaction_entry.get("task_id", i + 1)

        gt_entry = None
        for gt in ground_truth:
            if gt.get("task_id") == task_id:
                gt_entry = gt
                break

        if gt_entry is None:
            results.append({
                "scenario_type": scenario_type,
                "scenario_number": scenario_number,
                "task_id": task_id,
                "error_type": "missing_gt",
                "error_desc": "Matching ground truth record not found"
            })
            continue

        eval_detail = None
        for detail in eval_result.get("detailed_results", []):
            if detail.get("task_id") == task_id:
                eval_detail = detail
                break

        if eval_detail is None:
            results.append({
                "scenario_type": scenario_type,
                "scenario_number": scenario_number,
                "task_id": task_id,
                "error_type": "missing_eval",
                "error_desc": "Matching evaluation record not found"
            })
            continue

        results.append(analyze_scenario(scenario_type, gt_entry, interaction_entry, eval_detail, scenario_number))

    return {
        "model": model_name,
        "scenario_type": scenario_type,
        "scenario": scenario_number,
        "mode": mode,
        "results": results
    }


def calculate_error_rates(model_stats: Dict, model: str) -> Dict[str, float]:
    """
    Calculate proportions of each error type

    1. Syntax format error rate = syntax format error samples / total samples
    2. Multimodal recognition error rate = multimodal error samples / syntax-correct samples
    3. Hallucination error rate = incorrect user_id samples / syntax-correct samples
    4. Logic reasoning error rate = logic error samples / samples without the first three errors
    5. Memory/common sense error rate = memory error samples / samples without the first four errors
    6. Over-operation error rate = over-operation error samples / tool-based correct samples
    """
    stats = model_stats[model]
    total = stats["total"]

    if total == 0:
        return {}

    syntax_correct = total - stats["syntax_error"]
    no_multimodal_error = syntax_correct - stats["multimodal_error"]
    no_hallucination = no_multimodal_error - stats["hallucination_error"]
    no_logic_error = no_hallucination - stats["logic_error"]
    tool_correct = stats["correct"] + stats["over_operation_error"]

    rates = {
        "syntax_error_rate": stats["syntax_error"] / total if total > 0 else 0,
        "multimodal_error_rate": stats["multimodal_error"] / syntax_correct if syntax_correct > 0 else 0,
        "hallucination_rate": stats["hallucination_error"] / syntax_correct if syntax_correct > 0 else 0,
        "logic_reasoning_error_rate": stats["logic_error"] / no_hallucination if no_hallucination > 0 else 0,
        "memory_common_sense_error_rate": stats["memory_error"] / no_logic_error if no_logic_error > 0 else 0,
        "over_operation_error_rate": stats["over_operation_error"] / tool_correct if tool_correct > 0 else 0,
        "denominators": {
            "total": total,
            "syntax_correct": syntax_correct,
            "no_multimodal_error": no_multimodal_error,
            "no_hallucination": no_hallucination,
            "no_logic_error": no_logic_error,
            "tool_correct": tool_correct
        }
    }

    return rates


def generate_report(all_results: List[Dict]) -> str:
    """Generate error analysis report"""
    report_lines = []
    report_lines.append("=" * 120)
    report_lines.append("All-Scenario Error Reason Analysis Report (Based on eval_result Data)")
    report_lines.append("=" * 120)
    report_lines.append("")

    model_stats = defaultdict(lambda: {
        "total": 0,
        "syntax_error": 0,
        "multimodal_error": 0,
        "hallucination_error": 0,
        "logic_error": 0,
        "memory_error": 0,
        "over_operation_error": 0,
        "correct": 0
    })

    scenario_model_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "total": 0,
        "syntax_error": 0,
        "multimodal_error": 0,
        "hallucination_error": 0,
        "logic_error": 0,
        "memory_error": 0,
        "over_operation_error": 0,
        "correct": 0
    })))

    for result in all_results:
        if "error" in result:
            continue

        model = result["model"]
        scenario_type = result["scenario_type"]
        scenario = result["scenario"]

        for scenario_result in result["results"]:
            if scenario_result.get("error_type") in ["missing_gt", "missing_eval"]:
                continue

            model_stats[model]["total"] += 1
            scenario_model_stats[scenario_type][scenario][model]["total"] += 1

            error_type = scenario_result.get("error_type")
            if error_type and error_type in model_stats[model]:
                model_stats[model][error_type] += 1
            if error_type and error_type in scenario_model_stats[scenario_type][scenario][model]:
                scenario_model_stats[scenario_type][scenario][model][error_type] += 1

    report_lines.append("I. Overall Error Type Distribution (by Model) - Raw Counts")
    report_lines.append("-" * 120)
    report_lines.append("")

    header = f"{'Model':<30} {'Total':>6} {'Syntax':>6} {'MultiMod':>8} {'Halluc':>8} {'Logic':>8} {'Memory':>8} {'OverOp':>10} {'Correct':>8}"
    report_lines.append(header)
    report_lines.append("-" * 120)

    for model in MODELS:
        if model not in model_stats:
            continue

        stats = model_stats[model]
        total = stats["total"]
        if total == 0:
            continue

        row = f"{model:<30} {total:>6}"
        for error_type in [
            "syntax_error", "multimodal_error", "hallucination_error",
            "logic_error", "memory_error", "over_operation_error", "correct"
        ]:
            row += f" {stats[error_type]:>8}"
        report_lines.append(row)

    report_lines.append("")
    report_lines.append("II. Error Type Rate Statistics (using defined denominators)")
    report_lines.append("-" * 120)
    report_lines.append("")

    for model in MODELS:
        if model not in model_stats:
            continue

        stats = model_stats[model]
        if stats["total"] == 0:
            continue

        rates = calculate_error_rates(model_stats, model)
        denom = rates.get("denominators", {})

        report_lines.append(f"\nModel: {model}")
        report_lines.append("-" * 80)
        report_lines.append(f"1. Syntax format error rate: {stats['syntax_error']}/{denom.get('total', 0)} = {rates.get('syntax_error_rate', 0):.2%}")

        if denom.get('syntax_correct', 0) > 0:
            report_lines.append(f"2. Multimodal recognition error rate: {stats['multimodal_error']}/{denom.get('syntax_correct', 0)} = {rates.get('multimodal_error_rate', 0):.2%}")
        else:
            report_lines.append("2. Multimodal recognition error rate: N/A (no syntax-correct samples)")

        if denom.get('syntax_correct', 0) > 0:
            report_lines.append(f"3. Hallucination (user_id) error rate: {stats['hallucination_error']}/{denom.get('syntax_correct', 0)} = {rates.get('hallucination_rate', 0):.2%}")
        else:
            report_lines.append("3. Hallucination (user_id) error rate: N/A (no syntax-correct samples)")

        no_hallucination = denom.get('no_hallucination', 0)
        if no_hallucination > 0:
            report_lines.append(f"4. Logic reasoning error rate: {stats['logic_error']}/{no_hallucination} = {rates.get('logic_reasoning_error_rate', 0):.2%}")
        else:
            report_lines.append("4. Logic reasoning error rate: N/A (no samples correct for first three checks)")

        no_logic_error = denom.get('no_logic_error', 0)
        if no_logic_error > 0:
            report_lines.append(f"5. Memory/common sense error rate: {stats['memory_error']}/{no_logic_error} = {rates.get('memory_common_sense_error_rate', 0):.2%}")
        else:
            report_lines.append("5. Memory/common sense error rate: N/A (no samples correct for first four checks)")

        tool_correct = denom.get('tool_correct', 0)
        if tool_correct > 0:
            report_lines.append(f"6. Over-operation error rate: {stats['over_operation_error']}/{tool_correct} = {rates.get('over_operation_error_rate', 0):.2%}")
        else:
            report_lines.append("6. Over-operation error rate: N/A (no tool-call-correct samples)")

    report_lines.append("")
    report_lines.append("III. Error Type Distribution by Scenario (by Model)")
    report_lines.append("-" * 120)

    for scenario_type in ALL_SCENARIO_TYPES:
        if scenario_type not in scenario_model_stats:
            continue

        report_lines.append("")
        report_lines.append(f"Scenario Type: {scenario_type}")
        report_lines.append("-" * 100)

        for scenario in sorted(scenario_model_stats[scenario_type].keys()):
            report_lines.append("")
            report_lines.append(f"  Scenario {scenario}:")
            header = f"{'Model':<30} {'Total':>6} {'Syntax':>6} {'MultiM':>7} {'Halluc':>7} {'Logic':>7} {'Memory':>7} {'OverOp':>10} {'Correct':>7}"
            report_lines.append(header)

            for model in MODELS:
                if model not in scenario_model_stats[scenario_type][scenario]:
                    continue

                stats = scenario_model_stats[scenario_type][scenario][model]
                total = stats["total"]
                if total == 0:
                    continue

                row = f"{model:<30} {total:>6}"
                for error_type in [
                    "syntax_error", "multimodal_error", "hallucination_error",
                    "logic_error", "memory_error", "over_operation_error", "correct"
                ]:
                    row += f" {stats[error_type]:>8}"
                report_lines.append(row)

    report_lines.append("")
    report_lines.append("IV. Error Type Summary Across All Models")
    report_lines.append("-" * 120)
    report_lines.append("")

    total_all = sum(stats["total"] for stats in model_stats.values())
    error_type_totals = {
        "syntax_error": 0,
        "multimodal_error": 0,
        "hallucination_error": 0,
        "logic_error": 0,
        "memory_error": 0,
        "over_operation_error": 0,
        "correct": 0
    }

    for stats in model_stats.values():
        for error_type in error_type_totals:
            error_type_totals[error_type] += stats[error_type]

    syntax_correct_all = total_all - error_type_totals["syntax_error"]
    no_hallucination_all = (
        syntax_correct_all
        - error_type_totals["multimodal_error"]
        - error_type_totals["hallucination_error"]
    )
    no_logic_error_all = no_hallucination_all - error_type_totals["logic_error"]
    tool_correct_all = error_type_totals["correct"] + error_type_totals["over_operation_error"]

    report_lines.append("Error type counts and proportions:")
    report_lines.append("")
    report_lines.append(
        f"1. Syntax format error: {error_type_totals['syntax_error']}/{total_all} = {error_type_totals['syntax_error'] / total_all:.2%}"
        if total_all > 0 else "1. Syntax format error: N/A"
    )

    if syntax_correct_all > 0:
        report_lines.append(f"2. Multimodal recognition error: {error_type_totals['multimodal_error']}/{syntax_correct_all} = {error_type_totals['multimodal_error'] / syntax_correct_all:.2%}")
        report_lines.append(f"3. Hallucination (user_id): {error_type_totals['hallucination_error']}/{syntax_correct_all} = {error_type_totals['hallucination_error'] / syntax_correct_all:.2%}")
    else:
        report_lines.append("2. Multimodal recognition error: N/A")
        report_lines.append("3. Hallucination (user_id): N/A")

    if no_hallucination_all > 0:
        report_lines.append(f"4. Logic reasoning error: {error_type_totals['logic_error']}/{no_hallucination_all} = {error_type_totals['logic_error'] / no_hallucination_all:.2%}")
    else:
        report_lines.append("4. Logic reasoning error: N/A")

    if no_logic_error_all > 0:
        report_lines.append(f"5. Memory/common sense error: {error_type_totals['memory_error']}/{no_logic_error_all} = {error_type_totals['memory_error'] / no_logic_error_all:.2%}")
    else:
        report_lines.append("5. Memory/common sense error: N/A")

    if tool_correct_all > 0:
        report_lines.append(f"6. Over-operation error: {error_type_totals['over_operation_error']}/{tool_correct_all} = {error_type_totals['over_operation_error'] / tool_correct_all:.2%}")
    else:
        report_lines.append("6. Over-operation error: N/A")

    report_lines.append("")
    report_lines.append(
        f"Correct samples: {error_type_totals['correct']}/{total_all} = {error_type_totals['correct'] / total_all:.2%}"
        if total_all > 0 else "Correct samples: N/A"
    )
    report_lines.append("")
    report_lines.append(f"Total: {total_all} samples")
    report_lines.append("")
    report_lines.append("=" * 120)

    return "\n".join(report_lines)


def main():
    """Main function"""
    print("Starting all-scenario error reason analysis...")
    print("")

    all_results = []

    for scenario_type in ALL_SCENARIO_TYPES:
        print(f"Processing scenario type {scenario_type}...")

        for scenario_number in SCENARIO_CONFIG[scenario_type]['scenario_numbers']:
            print(f"  Processing scenario {scenario_number}...")

            ground_truth = load_ground_truth(scenario_type, scenario_number)
            if not ground_truth:
                print("    Skipped: unable to load ground truth file")
                continue

            for mode in MODES:
                for model in MODELS:
                    eval_result = load_eval_result(model, scenario_type, scenario_number, mode)
                    if eval_result is None:
                        continue

                    result = analyze_model_performance(
                        model, scenario_type, scenario_number, mode, ground_truth, eval_result
                    )
                    all_results.append(result)

    print("")
    print("Generating analysis report...")
    report = generate_report(all_results)
    print(report)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'error_analysis')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'all_scenarios_error_analysis_report.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")

    json_path = os.path.join(output_dir, 'all_scenarios_error_analysis_details.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to: {json_path}")


if __name__ == "__main__":
    main()
