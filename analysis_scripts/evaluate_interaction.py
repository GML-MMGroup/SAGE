import json
import hashlib
from typing import Any, Dict, List
from collections import Counter
import argparse
import inspect

# 1. Import database classes
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.retail.retail_db import RetailDB
from tools.retail.retail_init import retail_init_data1, retail_init_data2, retail_init_data3, retail_init_data4, retail_init_data5, retail_init_data6, retail_init_data7, retail_init_data8, retail_init_data9, retail_init_data10
from tools.kitchen.kitchen_db import KitchenDB
from tools.kitchen.kitchen_init import kitchen_init_data
from tools.restaurant.restaurant_db import RestaurantDB
from tools.restaurant.restaurant_init import restaurant_init_data, restaurant_init_data5
from tools.order.order_db import OrderDB
from tools.order.order_init import order_init_data

# ===================== Core Configuration: Fuzzy Match Fields & Scenario Mapping =====================
FUZZY_KEYS = {
    "retail": ["product_name"],
    "kitchen": ["ingredient_name", "recipe_name", "recipes"],
    "restaurant": ["dish_name", "set_meal_name"],
    "order": ["dish_name", "restaurant_name", "set_meal_name"]
}
# Database match method mapping
DB_MATCH_METHOD = {
    "retail": "_find_matching_products",
    "kitchen": None,  # Kitchen uses exact matching, not fuzzy matching
    "restaurant": "_find_matching_dishes",
    "order": "_find_matching_dishes"
}

# Set meal match method mapping (dish_name may be a set meal name, need to match both dishes and set meals)
DB_SET_MEAL_MATCH_METHOD = {
    "retail": None,
    "kitchen": None,
    "restaurant": "_find_matching_set_meals",
    "order": "_find_matching_set_meals"
}

# Scenario fuzzy match field config (for fuzzy matching within evaluate_interaction.py)
SCENARIO_FUZZY_FIELDS = {
    "retail": ["product_name"],
    "kitchen": ["ingredient_name", "recipe_name"],
    "restaurant": ["dish_name"],
    "order": ["dish_name", "restaurant_name"]
}


# ===================== Database Hash Calculation =====================
def calculate_db_hash(db_instance):
    """Calculate database state hash, supporting all scenario databases"""
    db_data = {}
    if isinstance(db_instance, KitchenDB):
        db_data = {
            'ingredients': {k: {'name': v.name, 'quantity': v.quantity, 'category': v.category,
                               'storage_location': v.storage_location, 'expiry_date': v.expiry_date,
                               'nutrition': vars(v.nutrition) if v.nutrition else None}
                         for k, v in db_instance.ingredients.items()},
            'recipes': {k: {'name': v.name, 'ingredients': [{'ingredient_name': i.ingredient_name, 'quantity': i.quantity} for i in v.ingredients],
                           'allergens': v.allergens, 'taste': v.taste, 'nutritional_characteristics': v.nutritional_characteristics}
                       for k, v in db_instance.recipes.items()},
            'user_menus': db_instance.user_menus,
            'user_shopping_lists': {k: sorted([{'ingredient_name': item.ingredient_name, 'quantity': item.quantity} for item in v.values()],
                                              key=lambda x: x['ingredient_name'])
                                   for k, v in db_instance.user_shopping_lists.items()}
        }
    elif isinstance(db_instance, RetailDB):
        db_data = {
            'catalog': {k: {'name': v.name, 'category': v.category, 'price': v.price, 'tax_rate': v.tax_rate,
                           'discount': v.discount, 'nutritional_characteristics': v.nutritional_characteristics,
                           'taste': v.taste, 'country_of_origin': v.country_of_origin,
                           'nutrition': vars(v.nutrition) if v.nutrition else None}
                     for k, v in db_instance.catalog.items()},
            'user_carts': {k: sorted([{'product_name': item.product_name, 'quantity': item.quantity,
                                       'category': item.category, 'price': item.price, 'tax_rate': item.tax_rate,
                                       'discount': item.discount} for item in v.values()],
                                      key=lambda x: x['product_name'])
                          for k, v in db_instance.user_carts.items()},
            'user_shopping_lists': db_instance.user_shopping_lists
        }
    elif isinstance(db_instance, RestaurantDB):
        db_data = {
            'catalog': {k: {'name': v.name, 'category': v.category, 'price': v.price, 'tax_rate': v.tax_rate,
                           'discount': v.discount, 'nutritional_characteristics': v.nutritional_characteristics,
                           'taste': v.taste, 'allergens': v.allergens,
                           'nutrition': vars(v.nutrition) if v.nutrition else None}
                     for k, v in db_instance.catalog.items()},
            'set_meals': {k: {'name': v.name, 'included_dishes': v.included_dishes,
                             'set_meal_price': v.set_meal_price, 'set_meal_discount': v.set_meal_discount}
                         for k, v in db_instance.set_meals.items()},
            'user_orders': {k: sorted([{'dish_name': item.dish_name, 'quantity': item.quantity} for item in v.values()],
                                       key=lambda x: x['dish_name'])
                           for k, v in db_instance.user_orders.items()}
        }
    elif isinstance(db_instance, OrderDB):
        db_data = {
            'restaurants': {}
        }
        for r_name, store in db_instance.restaurants.items():
            db_data['restaurants'][r_name] = {
                'catalog': {k: {'name': v.name, 'category': v.category, 'price': v.price, 'tax_rate': v.tax_rate,
                               'discount': v.discount, 'nutritional_characteristics': v.nutritional_characteristics,
                               'taste': v.taste, 'allergens': v.allergens,
                               'nutrition': vars(v.nutrition) if v.nutrition else None}
                         for k, v in store['catalog'].items()},
                'set_meals': {k: {'name': v.name, 'included_dishes': v.included_dishes,
                                 'set_meal_price': v.set_meal_price, 'set_meal_discount': v.set_meal_discount}
                             for k, v in store['set_meals'].items()},
                'user_orders': {k: sorted([{'dish_name': item.dish_name, 'quantity': item.quantity} for item in v.values()],
                                           key=lambda x: x['dish_name'])
                               for k, v in store['user_orders'].items()}
            }
    elif hasattr(db_instance, 'get_all_data'):
        db_data = db_instance.get_all_data()
    else:
        # Generic extraction
        for attr in dir(db_instance):
            if not attr.startswith('_') and not callable(getattr(db_instance, attr)):
                attr_value = getattr(db_instance, attr)
                try:
                    json.dumps(attr_value, sort_keys=True, default=str)
                    db_data[attr] = attr_value
                except:
                    continue

    json_str = json.dumps(db_data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


# ===================== Ground Truth Tool Call Simplification =====================
def simplify_tool_calls(db_instance, tool_calls):
    """
    Simplify ground truth tool calls, keeping only parameters needed by database methods
    Use actual database method signatures as reference, discard extra parameters
    """
    simplified_calls = []
    for tool_call in tool_calls:
        try:
            method_name = tool_call.get("tool_name") or tool_call.get("name")
            params = tool_call.get("parameters", {})

            if hasattr(db_instance, method_name):
                method = getattr(db_instance, method_name)
                sig = inspect.signature(method)

                # Keep only parameters accepted by the method signature
                valid_params = {
                    k: v for k, v in params.items()
                    if k in sig.parameters
                }

                simplified_calls.append({
                    "tool_name": method_name,
                    "parameters": valid_params
                })
            else:
                # If method does not exist, keep original call (for subsequent error statistics)
                simplified_calls.append(tool_call)
        except Exception as e:
            # Keep original call on error
            simplified_calls.append(tool_call)

    return simplified_calls


# ===================== Tool Execution Function =====================
def execute_tool_chain(db_instance, tool_calls):
    """Execute tool call chain (generic adapter) - with parameter filtering, keeping only method-accepted parameters"""
    results = []
    for tool_call in tool_calls:
        try:
            method_name = tool_call.get("tool_name") or tool_call.get("name")
            params = tool_call.get("parameters", {})

            if hasattr(db_instance, method_name):
                method = getattr(db_instance, method_name)

                # Filter parameters, keeping only those in the method signature
                sig = inspect.signature(method)
                valid_params = {
                    k: v for k, v in params.items()
                    if k in sig.parameters
                }

                result = method(**valid_params)
                results.append({
                    "tool_name": method_name,
                    "parameters": valid_params,
                    "result": result,
                    "status": "success"
                })
            else:
                results.append({
                    "tool_name": method_name,
                    "parameters": params,
                    "result": f"Tool '{method_name}' not found",
                    "status": "error"
                })
        except Exception as e:
            results.append({
                "tool_name": tool_call.get("tool_name") or tool_call.get("name"),
                "parameters": tool_call.get("parameters", {}),
                "result": str(e),
                "status": "error"
            })
    return results


# ===================== Fuzzy Match Function =====================
def fuzzy_match_str(query: str, target: str) -> bool:
    """Generic string fuzzy matching: lowercase containment check"""
    if not query or not target:
        return False
    return query.lower() in target.lower() or target.lower() in query.lower()


# ===================== Recursive Parameter Comparison =====================
def compare_parameters_recursive(
    gt_val: Any,
    inter_val: Any,
    db_instance: Any = None,
    scenario: str = "retail",
    current_key: str = None
) -> bool:
    """
    Recursively compare two parameter values, supporting:
    1. Array/list: unordered comparison, ignoring order
    2. Dict/object: recursive field-by-field comparison
    3. Specified fields: scenario-based fuzzy matching
    4. Basic types: exact comparison
    """
    # When types differ, try numeric-compatible comparison (str/int/float mix, e.g. "2" vs 2.0)
    if type(gt_val) != type(inter_val):
        try:
            if isinstance(gt_val, (str, int, float)) and isinstance(inter_val, (str, int, float)):
                return float(gt_val) == float(inter_val)
        except (ValueError, TypeError):
            pass
        return False

    # 1. Handle array/list type: unordered comparison
    if isinstance(gt_val, list):
        if len(gt_val) != len(inter_val):
            return False

        gt_matched = [False] * len(gt_val)
        inter_matched = [False] * len(inter_val)

        for i, gt_item in enumerate(gt_val):
            for j, inter_item in enumerate(inter_val):
                if not inter_matched[j] and compare_parameters_recursive(gt_item, inter_item, db_instance, scenario, current_key):
                    gt_matched[i] = True
                    inter_matched[j] = True
                    break
        return all(gt_matched)

    # 2. Handle dict/object type
    if isinstance(gt_val, dict):
        if set(gt_val.keys()) != set(inter_val.keys()):
            return False

        for key, g_val in gt_val.items():
            i_val = inter_val[key]
            # Check if current field needs fuzzy matching
            if key in FUZZY_KEYS.get(scenario, []):
                # If value is string, use fuzzy matching directly
                if isinstance(g_val, str):
                    if not fuzzy_match_field(g_val, i_val, db_instance, scenario):
                        return False
                else:
                    # If value is list or other type, continue recursive comparison (keep current key for subsequent string comparison)
                    if not compare_parameters_recursive(g_val, i_val, db_instance, scenario, key):
                        return False
            else:
                if not compare_parameters_recursive(g_val, i_val, db_instance, scenario, key):
                    return False
        return True

    # 3. Basic types: special string handling
    if isinstance(gt_val, str):
        # If current field is in FUZZY_KEYS, use case-insensitive comparison
        if current_key in FUZZY_KEYS.get(scenario, []):
            return gt_val.lower().strip() == inter_val.lower().strip()
        return gt_val == inter_val

    # 4. Other basic types: exact comparison directly
    return gt_val == inter_val


# ===================== Scenario-based Fuzzy Matching Core Function =====================
def fuzzy_match_field(gt_name: str, inter_name: str, db_instance: Any, scenario: str) -> bool:
    """
    Perform fuzzy matching based on scenario; dish_name matches both dishes and set meals, either match counts as correct
    """
    # No database instance: fallback to string fuzzy matching
    if not db_instance:
        return fuzzy_match_str(inter_name, gt_name)

    if scenario == "kitchen":
        # Kitchen scenario: use exact matching (KitchenDB has no fuzzy match method)
        return gt_name.lower().strip() == inter_name.lower().strip()

    match_method = DB_MATCH_METHOD.get(scenario)
    set_meal_method = DB_SET_MEAL_MATCH_METHOD.get(scenario)

    def _collect_matching_names(name: str) -> set:
        """Collect all matching names for dishes and set meals"""
        all_names = set()

        # Dish matching
        if match_method and hasattr(db_instance, match_method):
            try:
                match_func = getattr(db_instance, match_method)
                if scenario == "order":
                    # OrderDB needs to iterate over all restaurants
                    for r_name in db_instance.restaurants:
                        matches = match_func(r_name, name)
                        all_names.update(m.name for m in matches)
                else:
                    matches = match_func(name)
                    all_names.update(m.name for m in matches)
            except:
                pass

        # Set meal matching
        if set_meal_method and hasattr(db_instance, set_meal_method):
            try:
                sm_func = getattr(db_instance, set_meal_method)
                if scenario == "order":
                    for r_name in db_instance.restaurants:
                        matches = sm_func(r_name, name)
                        all_names.update(m.name for m in matches)
                else:
                    matches = sm_func(name)
                    all_names.update(m.name for m in matches)
            except:
                pass

        return all_names

    gt_names = _collect_matching_names(gt_name)
    inter_names = _collect_matching_names(inter_name)

    # Match succeeds if dishes or set meals have intersection
    if len(gt_names) > 0 and len(inter_names) > 0:
        return len(gt_names & inter_names) > 0

    # Database methods all returned empty lists, fallback to string fuzzy matching
    return fuzzy_match_str(inter_name, gt_name)


# ===================== Parameter Comparison Wrapper =====================
def compare_parameters_with_fuzzy_match(
    gt_params: Dict[str, Any],
    interaction_params: Dict[str, Any],
    db_instance: Any = None,
    scenario: str = "retail"
) -> bool:
    """Upper-level wrapper: compatible with original function parameters, added scenario parameter"""
    return compare_parameters_recursive(gt_params, interaction_params, db_instance, scenario)


# ===================== Tool Call Comparison =====================
def compare_tool_calls(ground_truth_calls, interaction_calls, db_instance=None, scenario="retail"):
    """
    Compare tool calls:
    1. Tool name exact match
    2. Parameters: unordered arrays + scenario-based fuzzy matching
    3. Overall call sequence unordered comparison
    4. Parameter filtering: only compare parameters needed by database methods
    """
    def extract_call_info(call):
        if isinstance(call, dict):
            return {
                "tool_name": call.get("tool_name") or call.get("name"),  # Support both field names
                "parameters": call.get("parameters", {})
            }
        return call

    def filter_params_by_method(tool_name, params, db):
        """Filter parameters based on database method signature"""
        if db and hasattr(db, tool_name):
            try:
                method = getattr(db, tool_name)
                sig = inspect.signature(method)
                return {k: v for k, v in params.items() if k in sig.parameters}
            except:
                pass
        return params

    try:
        # Ground truth calls have been simplified in the main evaluation function, use directly here
        gt_calls = [extract_call_info(call) for call in ground_truth_calls]

        # Extract model calls and filter parameters
        # New format: {"turn": 0, "calls": [...], "results": [...]}
        # Old format: {"turn": 0, "call": {...}, "result": "..."}
        interaction_only_calls = []
        for entry in interaction_calls:
            if isinstance(entry, dict):
                # New format: use "calls" list
                if "calls" in entry and isinstance(entry["calls"], list):
                    for call in entry["calls"]:
                        call_info = extract_call_info(call)
                        call_info["parameters"] = filter_params_by_method(
                            call_info["tool_name"],
                            call_info["parameters"],
                            db_instance
                        )
                        interaction_only_calls.append(call_info)
                # Old format: use "call" single object (backward compatible)
                elif "call" in entry:
                    call_info = extract_call_info(entry["call"])
                    call_info["parameters"] = filter_params_by_method(
                        call_info["tool_name"],
                        call_info["parameters"],
                        db_instance
                    )
                    interaction_only_calls.append(call_info)

        matches = 0
        matched_interaction_indices = set()

        for gt_call in gt_calls:
            for idx, interaction_call in enumerate(interaction_only_calls):
                if idx in matched_interaction_indices:
                    continue

                if gt_call.get("tool_name") == interaction_call.get("tool_name"):
                    if compare_parameters_with_fuzzy_match(
                        gt_call.get("parameters", {}),
                        interaction_call.get("parameters", {}),
                        db_instance,
                        scenario
                    ):
                        matches += 1
                        matched_interaction_indices.add(idx)
                        break

        return matches, len(gt_calls), len(interaction_only_calls)
    except:
        return 0, 0, 0


# ===================== Database Initialization =====================
def get_init_db(scenario, scenario_number):
    import io
    import sys

    # Temporarily redirect print output to avoid database initialization messages
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        db = None
        if scenario == "retail":
            db = RetailDB()
            init_data = [
                retail_init_data1, retail_init_data2, retail_init_data3, retail_init_data4,
                retail_init_data5, retail_init_data6, retail_init_data7, retail_init_data8,
                retail_init_data9, retail_init_data10
            ]
            if 1 <= scenario_number <= 10:
                db.init_from_json(init_data[scenario_number-1])
        elif scenario == "kitchen":
            db = KitchenDB()
            db.init_from_json(kitchen_init_data)
        elif scenario == "restaurant":
            db = RestaurantDB()
            if 1 <= scenario_number <= 4:
                db.init_from_json(restaurant_init_data)
            elif scenario_number == 5:
                db.init_from_json(restaurant_init_data5)
        elif scenario == "order":
            db = OrderDB()
            db.init_from_json(order_init_data)
        return db
    finally:
        sys.stdout = old_stdout


# ===================== Main Evaluation Function =====================
def evaluate_interaction_success(ground_truth_file, interaction_log_file, scenario="kitchen", args=None, silent=False, num_samples=0):
    """
    Evaluate interaction success rate

    Args:
        ground_truth_file: Ground truth file path
        interaction_log_file: Interaction log file path
        scenario: Scenario type (kitchen, retail, restaurant, order)
        args: Argument object containing scenario_number
        silent: Whether to run in silent mode
        num_samples: Number of samples per scenario to test, 0 means test all samples
    """
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    with open(interaction_log_file, 'r', encoding='utf-8') as f:
        interaction_data = json.load(f)

    # Limit sample count based on num_samples
    if num_samples > 0:
        interaction_data = interaction_data[:num_samples]

    gt_by_task_id = {
        gt.get("task_id", idx + 1): gt
        for idx, gt in enumerate(ground_truth_data)
    }

    results = {
        "total_scenarios": len(interaction_data),
        "valid_scenarios": 0,
        "invalid_scenarios": [],
        "tool_based": {"success_count": 0, "partial_matches": [], "success_rate": 0.0},
        "result_based": {"success_count": 0, "success_rate": 0.0},
        "joint_success": {"success_count": 0, "success_rate": 0.0},  # Joint success rate
        "detailed_results": [],
        "micro_tool_stats": {
            "total_correct_calls": 0, "total_ground_truth_calls": 0,
            "total_interaction_calls": 0, "micro_accuracy": 0.0,
            "task_count": 0  # Actual number of evaluated tasks (denominator)
        },
        "performance_metrics": {
            "avg_user_response_time": 0.0,
            "avg_agent_response_time": 0.0,
            "avg_tokens_consumed": 0.0,
            "avg_rounds_count": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "avg_tool_calls_count": 0.0,
            "avg_user_performance": {
                "original_role_consistency_avg": 0.0,
                "original_instruction_following_avg": 0.0,
                "original_resilience_avg": 0.0,
                "original_contextual_robustness_avg": 0.0,
                "final_role_consistency_avg": 0.0,
                "final_instruction_following_avg": 0.0,
                "final_resilience_avg": 0.0,
                "final_contextual_robustness_avg": 0.0
            }
        },
        # Scenario-level statistics
        "scenario_stats": {
            "per_scenario": [],  # Tool call count and token consumption per scenario
            "success_scenarios": {"total_tool_calls": 0, "total_tokens": 0, "count": 0},
            "failure_scenarios": {"total_tool_calls": 0, "total_tokens": 0, "count": 0}
        }
    }

    total_correct_calls = total_gt_calls = total_interaction_calls = 0
    total_user_response_time = 0.0
    total_agent_response_time = 0.0
    total_tokens_consumed = 0
    total_rounds_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tool_calls_count = 0
    total_user_performance = {
        "original_role_consistency_avg": 0.0,
        "original_instruction_following_avg": 0.0,
        "original_resilience_avg": 0.0,
        "original_contextual_robustness_avg": 0.0,
        "final_role_consistency_avg": 0.0,
        "final_instruction_following_avg": 0.0,
        "final_resilience_avg": 0.0,
        "final_contextual_robustness_avg": 0.0
    }
    valid_interaction_count = 0

    for interaction_idx, interaction_scenario in enumerate(interaction_data):
        task_id = interaction_scenario.get("task_id", interaction_idx + 1)
        gt_scenario = gt_by_task_id.get(task_id)
        if gt_scenario is None:
            if not silent:
                print(f"Warning: ground truth missing scenario {task_id} data, skipped")
            results["invalid_scenarios"].append({
                "task_id": task_id,
                "reason": "Missing ground truth data"
            })
            continue

        # Data format validation
        gt_tool_calls_raw = gt_scenario.get("ground_truth", [])
        interaction_tool_calls = interaction_scenario.get("tool_calls", [])
        if not isinstance(gt_tool_calls_raw, list) or not isinstance(interaction_tool_calls, list):
            results["invalid_scenarios"].append({
                "task_id": task_id,
                "reason": "Data format error (not a list)"
            })
            continue

        # Validation passed: include in valid evaluation
        valid_interaction_count += 1
        results["valid_scenarios"] += 1

        # Performance metrics statistics
        total_tokens_consumed += interaction_scenario.get("tokens_consumed", 0)
        total_user_response_time += interaction_scenario.get("user_response_time_seconds", 0.0)
        total_agent_response_time += interaction_scenario.get("agent_response_time_seconds", 0.0)
        total_rounds_count += interaction_scenario.get("rounds_count", 0)
        total_input_tokens += interaction_scenario.get("input_tokens", 0)
        total_output_tokens += interaction_scenario.get("output_tokens", 0)
        total_tool_calls_count += interaction_scenario.get("tool_calls_count", 0)

        user_perf = interaction_scenario.get("user_performance", {})
        if user_perf:
            for k in total_user_performance:
                total_user_performance[k] += user_perf.get(k, 0.0)

        detailed_result = {
            "task_id": task_id,
            "tool_based": {"success": False, "matches": 0, "total_gt_calls": 0, "total_interaction_calls": 0},
            "result_based": {"success": False, "gt_hash": None, "interaction_hash": None},
            "joint_success": False  # Joint success flag
        }

        # Get database instance for ground truth simplification and comparison
        db_instance_for_matching = get_init_db(scenario, args.scenario_number)

        # Simplify ground truth tool calls: keep only parameters needed by database methods
        gt_tool_calls = simplify_tool_calls(db_instance_for_matching, gt_tool_calls_raw)

        # Tool call evaluation (using simplified ground truth)
        matches, total_gt, total_interactions = compare_tool_calls(
            gt_tool_calls, interaction_tool_calls, db_instance_for_matching, scenario
        )
        detailed_result["tool_based"].update({"matches": matches, "total_gt_calls": total_gt, "total_interaction_calls": total_interactions})

        total_correct_calls += matches
        total_gt_calls += total_gt
        total_interaction_calls += total_interactions

        tool_success = False
        if matches == total_gt and total_gt > 0:
            results["tool_based"]["success_count"] += 1
            detailed_result["tool_based"]["success"] = True
            tool_success = True
        elif matches > 0:
            results["tool_based"]["partial_matches"].append({"task_id": task_id, "matches": matches, "total": total_gt})

        # Final result evaluation
        result_success = False
        try:
            gt_db = get_init_db(scenario, args.scenario_number)
            gt_tool_calls = execute_tool_chain(gt_db, gt_tool_calls)
            gt_hash = calculate_db_hash(gt_db)

            interaction_db = get_init_db(scenario, args.scenario_number)
            # New format: {"turn": 0, "calls": [...], "results": [...]}
            # Old format: {"turn": 0, "call": {...}, "result": "..."}
            interaction_only_calls = []
            for entry in interaction_tool_calls:
                if isinstance(entry, dict):
                    if "calls" in entry and isinstance(entry["calls"], list):
                        # New format: expand all calls
                        interaction_only_calls.extend(entry["calls"])
                    elif "call" in entry:
                        # Old format: backward compatible
                        interaction_only_calls.append(entry["call"])
            interaction_only_calls = execute_tool_chain(interaction_db, interaction_only_calls)
            interaction_hash = calculate_db_hash(interaction_db)

            detailed_result["result_based"].update({"gt_hash": gt_hash, "interaction_hash": interaction_hash})
            if gt_hash == interaction_hash:
                results["result_based"]["success_count"] += 1
                detailed_result["result_based"]["success"] = True
                result_success = True
        except Exception as e:
            if not silent:
                print(f"Scenario {task_id} database operation failed, skipped result evaluation: {e}")

        # Joint success: only when both process-based and result-based evaluations succeed
        if tool_success and result_success:
            results["joint_success"]["success_count"] += 1
            detailed_result["joint_success"] = True

        # Check if the simulated user response was corrected. New logs store this
        # per user turn under dialogue[*].evaluation; corrected_scores is kept as
        # a backward-compatible fallback for older result files.
        user_has_issue = any(
            msg.get("role") == "user"
            and msg.get("evaluation", {}).get("correction_applied", False)
            for msg in interaction_scenario.get("dialogue", [])
            if isinstance(msg, dict)
        )
        if not user_has_issue:
            corrected_scores = interaction_scenario.get("corrected_scores", {})
            if corrected_scores:
                for score_key in ["role_consistency", "instruction_following", "resilience", "contextual_robustness"]:
                    if corrected_scores.get(score_key, 1) == 0:
                        user_has_issue = True
                        break
        detailed_result["user_has_issue"] = user_has_issue

        # Scenario-level statistics: tool call count and token consumption
        scenario_tokens = interaction_scenario.get("tokens_consumed", 0)
        scenario_tool_calls = len(interaction_tool_calls)

        # Add to per_scenario list
        results["scenario_stats"]["per_scenario"].append({
            "task_id": task_id,
            "tool_calls": scenario_tool_calls,
            "tokens_consumed": scenario_tokens,
            "is_success": tool_success and result_success,
            "user_has_issue": user_has_issue
        })

        # Separate statistics for success and failure scenarios
        if tool_success and result_success:
            results["scenario_stats"]["success_scenarios"]["total_tool_calls"] += scenario_tool_calls
            results["scenario_stats"]["success_scenarios"]["total_tokens"] += scenario_tokens
            results["scenario_stats"]["success_scenarios"]["count"] += 1
        else:
            results["scenario_stats"]["failure_scenarios"]["total_tool_calls"] += scenario_tool_calls
            results["scenario_stats"]["failure_scenarios"]["total_tokens"] += scenario_tokens
            results["scenario_stats"]["failure_scenarios"]["count"] += 1

        results["detailed_results"].append(detailed_result)

    # Calculate rates
    if results["valid_scenarios"] > 0:
        results["tool_based"]["success_rate"] = results["tool_based"]["success_count"] / results["valid_scenarios"]
        results["result_based"]["success_rate"] = results["result_based"]["success_count"] / results["valid_scenarios"]
        results["joint_success"]["success_rate"] = results["joint_success"]["success_count"] / results["valid_scenarios"]

    # Calculate average tool calls and tokens for success/failure scenarios
    success_count = results["scenario_stats"]["success_scenarios"]["count"]
    failure_count = results["scenario_stats"]["failure_scenarios"]["count"]

    if success_count > 0:
        results["scenario_stats"]["success_scenarios"]["avg_tool_calls"] = \
            results["scenario_stats"]["success_scenarios"]["total_tool_calls"] / success_count
        results["scenario_stats"]["success_scenarios"]["avg_tokens"] = \
            results["scenario_stats"]["success_scenarios"]["total_tokens"] / success_count
    else:
        results["scenario_stats"]["success_scenarios"]["avg_tool_calls"] = 0.0
        results["scenario_stats"]["success_scenarios"]["avg_tokens"] = 0.0

    if failure_count > 0:
        results["scenario_stats"]["failure_scenarios"]["avg_tool_calls"] = \
            results["scenario_stats"]["failure_scenarios"]["total_tool_calls"] / failure_count
        results["scenario_stats"]["failure_scenarios"]["avg_tokens"] = \
            results["scenario_stats"]["failure_scenarios"]["total_tokens"] / failure_count
    else:
        results["scenario_stats"]["failure_scenarios"]["avg_tool_calls"] = 0.0
        results["scenario_stats"]["failure_scenarios"]["avg_tokens"] = 0.0

    # Micro tool call success rate: use actual evaluated task count as denominator
    task_count = len(results["detailed_results"])  # Actual number of evaluated tasks
    results["micro_tool_stats"].update({
        "total_correct_calls": total_correct_calls,
        "total_ground_truth_calls": total_gt_calls,
        "total_interaction_calls": total_interaction_calls,
        "task_count": task_count,  # Actual number of evaluated tasks
        "micro_accuracy": total_correct_calls / total_gt_calls if total_gt_calls > 0 else 0.0,
        # Per-task tool call accuracy (average)
        "avg_task_accuracy": sum(d["tool_based"]["matches"] / d["tool_based"]["total_gt_calls"]
                                  for d in results["detailed_results"]
                                  if d["tool_based"]["total_gt_calls"] > 0) / task_count if task_count > 0 else 0.0
    })

    if valid_interaction_count > 0:
        results["performance_metrics"]["avg_user_response_time"] = total_user_response_time / valid_interaction_count
        results["performance_metrics"]["avg_agent_response_time"] = total_agent_response_time / valid_interaction_count
        results["performance_metrics"]["avg_tokens_consumed"] = total_tokens_consumed / valid_interaction_count
        results["performance_metrics"]["avg_rounds_count"] = total_rounds_count / valid_interaction_count
        results["performance_metrics"]["avg_input_tokens"] = total_input_tokens / valid_interaction_count
        results["performance_metrics"]["avg_output_tokens"] = total_output_tokens / valid_interaction_count
        results["performance_metrics"]["avg_tool_calls_count"] = total_tool_calls_count / valid_interaction_count
        for k in total_user_performance:
            results["performance_metrics"]["avg_user_performance"][k] = total_user_performance[k] / valid_interaction_count

    # ===================== Statistics Excluding User-Issue Samples =====================
    filtered_details = [d for d in results["detailed_results"] if not d.get("user_has_issue", False)]
    user_issue_count = len(results["detailed_results"]) - len(filtered_details)
    filtered_valid = len(filtered_details)

    filtered_tool_success = sum(1 for d in filtered_details if d["tool_based"]["success"])
    filtered_result_success = sum(1 for d in filtered_details if d["result_based"]["success"])
    filtered_joint_success = sum(1 for d in filtered_details if d.get("joint_success", False))
    filtered_correct_calls = sum(d["tool_based"]["matches"] for d in filtered_details)
    filtered_gt_calls = sum(d["tool_based"]["total_gt_calls"] for d in filtered_details)

    results["filtered_user_issue"] = {
        "user_issue_count": user_issue_count,
        "total_valid_scenarios": results["valid_scenarios"],
        "user_issue_ratio": user_issue_count / results["valid_scenarios"] if results["valid_scenarios"] > 0 else 0.0,
        "filtered_valid_scenarios": filtered_valid,
        "filtered_tool_based_success_count": filtered_tool_success,
        "filtered_tool_based_success_rate": filtered_tool_success / filtered_valid if filtered_valid > 0 else 0.0,
        "filtered_result_based_success_count": filtered_result_success,
        "filtered_result_based_success_rate": filtered_result_success / filtered_valid if filtered_valid > 0 else 0.0,
        "filtered_joint_success_count": filtered_joint_success,
        "filtered_joint_success_rate": filtered_joint_success / filtered_valid if filtered_valid > 0 else 0.0,
        "filtered_micro_accuracy": filtered_correct_calls / filtered_gt_calls if filtered_gt_calls > 0 else 0.0,
        "filtered_avg_task_accuracy": sum(d["tool_based"]["matches"] / d["tool_based"]["total_gt_calls"]
                                          for d in filtered_details
                                          if d["tool_based"]["total_gt_calls"] > 0) / filtered_valid if filtered_valid > 0 else 0.0
    }

    return results


# ===================== Report Printing =====================
def print_evaluation_report(results):
    print("=" * 60)
    print("Interaction Success Rate Evaluation Report")
    print("=" * 60)
    print(f"Total scenarios: {results['total_scenarios']}")
    print(f"Valid evaluation scenarios: {results['valid_scenarios']}")
    print(f"Invalid skipped scenarios: {len(results['invalid_scenarios'])}")
    if results["invalid_scenarios"]:
        print("Skipped invalid samples:")
        for item in results["invalid_scenarios"]:
            print(f"  - Scenario {item['task_id']}: {item['reason']}")
    print()

    print("1. Tool-based evaluation:")
    print(f"   Successful scenarios: {results['tool_based']['success_count']}")
    print(f"   Success rate: {results['tool_based']['success_rate']:.2%}")
    if results['tool_based']['partial_matches']:
        print("   Partially matched scenarios:")
        for m in results['tool_based']['partial_matches']:
            print(f"     Scenario {m['task_id']}: {m['matches']}/{m['total']}")
    print()

    print("2. Result-based evaluation:")
    print(f"   Successful scenarios: {results['result_based']['success_count']}")
    print(f"   Success rate: {results['result_based']['success_rate']:.2%}\n")

    print("3. Joint success rate (process + result dual evaluation):")
    print(f"   Successful scenarios: {results['joint_success']['success_count']}")
    print(f"   Success rate: {results['joint_success']['success_rate']:.2%}\n")

    print("4. Per-task tool call detailed statistics:")
    for d in results['detailed_results']:
        print(f"   Scenario {d['task_id']}: correct {d['tool_based']['matches']}/{d['tool_based']['total_gt_calls']}")

    print("\n5. Scenario-level statistics:")
    scenario_stats = results.get("scenario_stats", {})
    per_scenario = scenario_stats.get("per_scenario", [])
    if per_scenario:
        print(f"   {'Scenario ID':<12} {'Tool Calls':<12} {'Tokens':<12} {'Status':<8}")
        print("   " + "-" * 44)
        for s in per_scenario:
            status = "Success" if s.get("is_success") else "Failure"
            print(f"   Scenario {s['task_id']:<8} {s['tool_calls']:<12} {s['tokens_consumed']:<12} {status:<8}")

    success_scenarios = scenario_stats.get("success_scenarios", {})
    failure_scenarios = scenario_stats.get("failure_scenarios", {})

    print("\n6. Success/failure scenario comparison:")
    print(f"   Success scenario count: {success_scenarios.get('count', 0)}")
    print(f"   Success scenario total tool calls: {success_scenarios.get('total_tool_calls', 0)}")
    print(f"   Success scenario avg tool calls: {success_scenarios.get('avg_tool_calls', 0):.2f}")
    print(f"   Success scenario avg tokens: {success_scenarios.get('avg_tokens', 0):.2f}")
    print()
    print(f"   Failure scenario count: {failure_scenarios.get('count', 0)}")
    print(f"   Failure scenario total tool calls: {failure_scenarios.get('total_tool_calls', 0)}")
    print(f"   Failure scenario avg tool calls: {failure_scenarios.get('avg_tool_calls', 0):.2f}")
    print(f"   Failure scenario avg tokens: {failure_scenarios.get('avg_tokens', 0):.2f}")

    print("\n7. Micro tool call success rate:")
    print(f"   Evaluated tasks: {results['micro_tool_stats']['task_count']}")
    print(f"   Total correct calls: {results['micro_tool_stats']['total_correct_calls']}")
    print(f"   Total ground truth calls: {results['micro_tool_stats']['total_ground_truth_calls']}")
    print(f"   Overall accuracy: {results['micro_tool_stats']['micro_accuracy']:.2%}")
    print(f"   Average task accuracy: {results['micro_tool_stats']['avg_task_accuracy']:.2%}")

    print("\n8. Additional performance metrics:")
    perf = results.get('performance_metrics', {})
    print(f"   Avg user response time: {perf.get('avg_user_response_time', 0):.2f} seconds")
    print(f"   Avg agent response time: {perf.get('avg_agent_response_time', 0):.2f} seconds")
    print(f"   Avg tokens consumed: {perf.get('avg_tokens_consumed', 0):.0f}")
    print(f"   Avg conversation rounds: {perf.get('avg_rounds_count', 0):.2f}")
    print(f"   Avg input tokens: {perf.get('avg_input_tokens', 0):.0f}")
    print(f"   Avg output tokens: {perf.get('avg_output_tokens', 0):.0f}")
    print(f"   Avg tool calls: {perf.get('avg_tool_calls_count', 0):.2f}")
    print("   Avg user performance:")
    user_perf_avg = perf.get("avg_user_performance", {})
    for k, v in user_perf_avg.items():
        print(f"     {k}: {v:.2f}")

    # Statistics excluding user-issue samples
    fui = results.get("filtered_user_issue", {})
    if fui:
        print(f"\n9. Statistics excluding inaccurate simulated user samples:")
        print(f"   User-issue samples: {fui.get('user_issue_count', 0)}/{fui.get('total_valid_scenarios', 0)} "
              f"(ratio: {fui.get('user_issue_ratio', 0):.2%})")
        print(f"   Filtered valid scenarios: {fui.get('filtered_valid_scenarios', 0)}")
        print(f"   Filtered tool success rate: {fui.get('filtered_tool_based_success_rate', 0):.2%}")
        print(f"   Filtered result success rate: {fui.get('filtered_result_based_success_rate', 0):.2%}")
        print(f"   Filtered joint success rate: {fui.get('filtered_joint_success_rate', 0):.2%}")
        print(f"   Filtered micro accuracy: {fui.get('filtered_micro_accuracy', 0):.2%}")
        print(f"   Filtered avg task accuracy: {fui.get('filtered_avg_task_accuracy', 0):.2%}")


# ===================== Main Function =====================
def main():
    import os
    import re

    parser = argparse.ArgumentParser(description="evaluation script")
    parser.add_argument("--model_name", type=str, default="gemini-3.1-pro-preview", help="Model name (subdirectory under results folder)")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples per scenario to test, 0 means test all samples")
    args = parser.parse_args()

    model_name = args.model_name
    num_samples = args.num_samples
    results_dir = f"../results/{model_name}"

    if not os.path.exists(results_dir):
        print(f"Error: result directory '{results_dir}' does not exist")
        return

    # Ensure output directory exists
    output_dir = f"../eval_result/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Get all JSON files recursively so results/{model}/{video_model}/{scenario}/file.json works.
    json_files = []
    for root, _, files in os.walk(results_dir):
        for file_name in files:
            if file_name.endswith('.json'):
                json_files.append(os.path.relpath(os.path.join(root, file_name), results_dir))

    if not json_files:
        print(f"Error: no JSON files found under directory '{results_dir}'")
        return

    print(f"Found {len(json_files)} result files, starting evaluation...\n")

    # Filename format: {YYYYMMDD}_{HHMMSS}_{scenario}{number}_{mode}.json
    # Scenario prefixes: kitchen, retail, restaurant, order
    scenario_prefixes = ["kitchen", "retail", "restaurant", "order"]

    # Aggregate all results
    all_results = []
    file_pattern = re.compile(r'^(\d{8}_\d{6})_([a-z]+)(\d+)_(easy|hard|static)\.json$')

    for json_file in sorted(json_files):
        json_name = os.path.basename(json_file)
        match = file_pattern.match(json_name)
        if not match:
            print(f"Skipping unparseable file: {json_file}")
            continue

        timestamp = match.group(1)
        scenario_prefix = match.group(2)
        scenario_number = int(match.group(3))
        user_mode = match.group(4)

        # Validate scenario prefix
        if scenario_prefix not in scenario_prefixes:
            print(f"Skipping file with unknown scenario prefix: {json_file}")
            continue

        ground_truth_file = f"../scenarios/final/{scenario_prefix}{scenario_number}.json"
        interaction_log_file = os.path.join(results_dir, json_file)

        # Check if ground truth file exists
        if not os.path.exists(ground_truth_file):
            print(f"Skipped: ground truth file not found '{ground_truth_file}'")
            continue

        print(f"Evaluating: {json_file}...", end=" ")

        try:
            results = evaluate_interaction_success(
                ground_truth_file,
                interaction_log_file,
                scenario=scenario_prefix,
                args=argparse.Namespace(scenario_number=scenario_number),
                silent=True,
                num_samples=num_samples
            )

            # Save per-file evaluation results
            output_file = os.path.join(output_dir, json_file.replace('.json', '_eval.json'))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"Evaluation results saved to {output_file}")

            # Print brief report (success rates only)
            fui = results.get("filtered_user_issue", {})
            print(f"Valid samples: {results['valid_scenarios']}/{results['total_scenarios']}, "
                  f"Tool success rate: {results['tool_based']['success_rate']:.2%}, "
                  f"Result success rate: {results['result_based']['success_rate']:.2%}, "
                  f"Joint success rate: {results['joint_success']['success_rate']:.2%}, "
                  f"User-issue samples: {fui.get('user_issue_count', 0)}, "
                  f"Filtered joint success rate: {fui.get('filtered_joint_success_rate', 0):.2%}")

            # Extract scenario-level statistics
            scenario_stats = results.get("scenario_stats", {})
            success_stats = scenario_stats.get("success_scenarios", {})
            failure_stats = scenario_stats.get("failure_scenarios", {})

            all_results.append({
                "file": json_file,
                "scenario": scenario_prefix,
                "scenario_number": scenario_number,
                "mode": user_mode,
                "total_scenarios": results["total_scenarios"],
                "valid_scenarios": results["valid_scenarios"],
                "tool_based_success_rate": results["tool_based"]["success_rate"],
                "result_based_success_rate": results["result_based"]["success_rate"],
                "joint_success_rate": results["joint_success"]["success_rate"],
                "micro_accuracy": results["micro_tool_stats"]["micro_accuracy"],
                "avg_task_accuracy": results["micro_tool_stats"]["avg_task_accuracy"],
                # Scenario-level statistics
                "success_scenario_count": success_stats.get("count", 0),
                "failure_scenario_count": failure_stats.get("count", 0),
                "success_avg_tool_calls": success_stats.get("avg_tool_calls", 0),
                "failure_avg_tool_calls": failure_stats.get("avg_tool_calls", 0),
                "success_avg_tokens": success_stats.get("avg_tokens", 0),
                "failure_avg_tokens": failure_stats.get("avg_tokens", 0),
                # Additional performance metrics
                "avg_rounds_count": results["performance_metrics"]["avg_rounds_count"],
                "avg_input_tokens": results["performance_metrics"]["avg_input_tokens"],
                "avg_output_tokens": results["performance_metrics"]["avg_output_tokens"],
                "avg_tool_calls_count": results["performance_metrics"]["avg_tool_calls_count"],
                # Statistics excluding user-issue samples
                "user_issue_count": fui.get("user_issue_count", 0),
                "user_issue_ratio": fui.get("user_issue_ratio", 0),
                "filtered_valid_scenarios": fui.get("filtered_valid_scenarios", 0),
                "filtered_tool_based_success_rate": fui.get("filtered_tool_based_success_rate", 0),
                "filtered_result_based_success_rate": fui.get("filtered_result_based_success_rate", 0),
                "filtered_joint_success_rate": fui.get("filtered_joint_success_rate", 0),
                "filtered_micro_accuracy": fui.get("filtered_micro_accuracy", 0),
                "filtered_avg_task_accuracy": fui.get("filtered_avg_task_accuracy", 0)
            })

        except Exception as e:
            print(f"Evaluation failed {json_file}: {e}")
            all_results.append({
                "file": json_file,
                "error": str(e)
            })

    # Print summary report
    print("\n" + "="*120)
    print("Evaluation Summary Report")
    print("="*120)
    if num_samples > 0:
        print(f"Sample limit: first {num_samples} samples per scenario")
    else:
        print("Sample limit: test all samples")
    print(f"{'File':<40} {'Scenario':<12} {'Num':<6} {'Mode':<8} {'Valid':<10} {'Tool Rate':<10} {'Result Rate':<10} {'Joint Rate':<10} {'User Issues':<10} {'Filt.Joint':<10}")
    print("-"*150)

    for r in all_results:
        if "error" in r:
            print(f"{r['file']:<40} Error: {r['error']}")
        else:
            valid_str = f"{r.get('valid_scenarios', '?')}/{r.get('total_scenarios', '?')}"
            user_issue_str = f"{r.get('user_issue_count', 0)}/{r.get('valid_scenarios', '?')}"
            print(f"{r['file']:<40} {r['scenario']:<12} {r['scenario_number']:<6} {r['mode']:<8} "
                  f"{valid_str:<10} "
                  f"{r['tool_based_success_rate']:>8.2%}   {r['result_based_success_rate']:>8.2%}   "
                  f"{r['joint_success_rate']:>8.2%}   {user_issue_str:<10} {r.get('filtered_joint_success_rate', 0):>8.2%}")

    # Calculate average success rates
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        total_valid = sum(r["valid_scenarios"] for r in valid_results)
        avg_tool_success = sum(r["tool_based_success_rate"] * r["valid_scenarios"] for r in valid_results) / total_valid
        avg_result_success = sum(r["result_based_success_rate"] * r["valid_scenarios"] for r in valid_results) / total_valid
        avg_joint_success = sum(r["joint_success_rate"] * r["valid_scenarios"] for r in valid_results) / total_valid
        avg_micro_accuracy = sum(r["micro_accuracy"] * r["valid_scenarios"] for r in valid_results) / total_valid
        avg_task_accuracy = sum(r["avg_task_accuracy"] * r["valid_scenarios"] for r in valid_results) / total_valid

        # Calculate avg tool calls and tokens for success/failure scenarios (weighted average)
        total_success_count = sum(r["success_scenario_count"] for r in valid_results)
        total_failure_count = sum(r["failure_scenario_count"] for r in valid_results)

        weighted_success_avg_tool_calls = sum(
            r["success_avg_tool_calls"] * r["success_scenario_count"] for r in valid_results
        ) / total_success_count if total_success_count > 0 else 0
        weighted_success_avg_tokens = sum(
            r["success_avg_tokens"] * r["success_scenario_count"] for r in valid_results
        ) / total_success_count if total_success_count > 0 else 0
        weighted_failure_avg_tool_calls = sum(
            r["failure_avg_tool_calls"] * r["failure_scenario_count"] for r in valid_results
        ) / total_failure_count if total_failure_count > 0 else 0
        weighted_failure_avg_tokens = sum(
            r["failure_avg_tokens"] * r["failure_scenario_count"] for r in valid_results
        ) / total_failure_count if total_failure_count > 0 else 0

        # Additional metrics: weighted average by scenario count
        total_scenario_count = sum(r["success_scenario_count"] + r["failure_scenario_count"] for r in valid_results)
        avg_rounds_count = sum(
            r["avg_rounds_count"] * (r["success_scenario_count"] + r["failure_scenario_count"]) for r in valid_results
        ) / total_scenario_count if total_scenario_count > 0 else 0
        avg_input_tokens = sum(
            r["avg_input_tokens"] * (r["success_scenario_count"] + r["failure_scenario_count"]) for r in valid_results
        ) / total_scenario_count if total_scenario_count > 0 else 0
        avg_output_tokens = sum(
            r["avg_output_tokens"] * (r["success_scenario_count"] + r["failure_scenario_count"]) for r in valid_results
        ) / total_scenario_count if total_scenario_count > 0 else 0
        avg_tool_calls_count = sum(
            r["avg_tool_calls_count"] * (r["success_scenario_count"] + r["failure_scenario_count"]) for r in valid_results
        ) / total_scenario_count if total_scenario_count > 0 else 0

        print("-"*120)
        print(f"Average success rates: tool-based={avg_tool_success:.2%}, result-based={avg_result_success:.2%}, joint={avg_joint_success:.2%}")
        print(f"Micro tool calls: overall accuracy={avg_micro_accuracy:.2%}, avg task accuracy={avg_task_accuracy:.2%}")
        print()
        print(f"Summary statistics (weighted average):")
        print(f"   Success scenario avg tool calls: {weighted_success_avg_tool_calls:.2f}, avg tokens: {weighted_success_avg_tokens:.2f}")
        print(f"   Failure scenario avg tool calls: {weighted_failure_avg_tool_calls:.2f}, avg tokens: {weighted_failure_avg_tokens:.2f}")
        print(f"   Avg conversation rounds: {avg_rounds_count:.2f}")
        print(f"   Avg input tokens: {avg_input_tokens:.0f}")
        print(f"   Avg output tokens: {avg_output_tokens:.0f}")
        print(f"   Avg tool calls: {avg_tool_calls_count:.2f}")

        # Summary excluding user-issue samples
        total_user_issue = sum(r.get("user_issue_count", 0) for r in valid_results)
        total_filtered_valid = sum(r.get("filtered_valid_scenarios", 0) for r in valid_results)
        if total_filtered_valid > 0:
            avg_filtered_tool_success = sum(
                r["filtered_tool_based_success_rate"] * r["filtered_valid_scenarios"] for r in valid_results
            ) / total_filtered_valid
            avg_filtered_result_success = sum(
                r["filtered_result_based_success_rate"] * r["filtered_valid_scenarios"] for r in valid_results
            ) / total_filtered_valid
            avg_filtered_joint_success = sum(
                r["filtered_joint_success_rate"] * r["filtered_valid_scenarios"] for r in valid_results
            ) / total_filtered_valid
            avg_filtered_micro_accuracy = sum(
                r["filtered_micro_accuracy"] * r["filtered_valid_scenarios"] for r in valid_results
            ) / total_filtered_valid
            avg_filtered_task_accuracy = sum(
                r["filtered_avg_task_accuracy"] * r["filtered_valid_scenarios"] for r in valid_results
            ) / total_filtered_valid
        else:
            avg_filtered_tool_success = avg_filtered_result_success = avg_filtered_joint_success = 0
            avg_filtered_micro_accuracy = avg_filtered_task_accuracy = 0

        print()
        print(f"Summary excluding inaccurate simulated user samples:")
        print(f"   Total user-issue samples: {total_user_issue}/{total_valid} (ratio: {total_user_issue / total_valid:.2%})")
        print(f"   Filtered total valid scenarios: {total_filtered_valid}")
        print(f"   Filtered avg success rates: tool-based={avg_filtered_tool_success:.2%}, "
              f"result-based={avg_filtered_result_success:.2%}, joint={avg_filtered_joint_success:.2%}")
        print(f"   Filtered micro accuracy: {avg_filtered_micro_accuracy:.2%}, avg task accuracy: {avg_filtered_task_accuracy:.2%}")

        # Per-scenario user issue statistics
        print()
        print(f"Per-scenario user-issue sample statistics:")
        for r in valid_results:
            print(f"   {r['file']:<40} User-issue samples: {r.get('user_issue_count', 0)}/{r.get('valid_scenarios', 0)} "
                  f"(ratio: {r.get('user_issue_ratio', 0):.2%})")

    # Save summary results
    summary_file = f"{output_dir}/summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({"all_results": all_results, "summary": {
            "total_files": len(all_results),
            "valid_files": len(valid_results),
            "num_samples": num_samples,
            "avg_tool_based_success_rate": avg_tool_success if valid_results else 0,
            "avg_result_based_success_rate": avg_result_success if valid_results else 0,
            "avg_joint_success_rate": avg_joint_success if valid_results else 0,
            "micro_accuracy": avg_micro_accuracy if valid_results else 0,
            "avg_task_accuracy": avg_task_accuracy if valid_results else 0,
            # Summary statistics
            "total_success_scenarios": total_success_count if valid_results else 0,
            "total_failure_scenarios": total_failure_count if valid_results else 0,
            "weighted_success_avg_tool_calls": weighted_success_avg_tool_calls if valid_results else 0,
            "weighted_success_avg_tokens": weighted_success_avg_tokens if valid_results else 0,
            "weighted_failure_avg_tool_calls": weighted_failure_avg_tool_calls if valid_results else 0,
            "weighted_failure_avg_tokens": weighted_failure_avg_tokens if valid_results else 0,
            # Additional metrics
            "avg_rounds_count": avg_rounds_count if valid_results else 0,
            "avg_input_tokens": avg_input_tokens if valid_results else 0,
            "avg_output_tokens": avg_output_tokens if valid_results else 0,
            "avg_tool_calls_count": avg_tool_calls_count if valid_results else 0,
            # Summary excluding user-issue samples
            "total_user_issue_count": total_user_issue if valid_results else 0,
            "filtered_valid_scenarios": total_filtered_valid if valid_results else 0,
            "filtered_tool_based_success_rate": avg_filtered_tool_success if valid_results else 0,
            "filtered_result_based_success_rate": avg_filtered_result_success if valid_results else 0,
            "filtered_joint_success_rate": avg_filtered_joint_success if valid_results else 0,
            "filtered_micro_accuracy": avg_filtered_micro_accuracy if valid_results else 0,
            "filtered_avg_task_accuracy": avg_filtered_task_accuracy if valid_results else 0
        }}, f, ensure_ascii=False, indent=2)
    print(f"\nSummary results saved to {summary_file}")


if __name__ == "__main__":
    main()
