import json
import hashlib
from typing import Any, Dict
import argparse
import inspect

# 1. Import database classes
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.retail.retail_db import RetailDB
from tools.retail.retail_init import (
    retail_init_data1, retail_init_data2, retail_init_data3, retail_init_data4, retail_init_data5,
    retail_init_data6, retail_init_data7, retail_init_data8, retail_init_data9, retail_init_data10
)
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
    "order": ["dish_name", "set_meal_name", "restaurant_name"],
}

DB_MATCH_METHOD = {
    "retail": "_find_matching_products",
    "kitchen": None,
    "restaurant": "_find_matching_dishes",
    "order": "_find_matching_dishes"
}

DB_SET_MEAL_MATCH_METHOD = {
    "retail": None,
    "kitchen": None,
    "restaurant": "_find_matching_set_meals",
    "order": "_find_matching_set_meals"
}

SCENARIO_FUZZY_FIELDS = {
    "retail": ["product_name"],
    "kitchen": ["ingredient_name", "recipe_name"],
    "restaurant": ["dish_name", "set_meal_name"],
    "order": ["dish_name", "set_meal_name", "restaurant_name"],
}

# ===================== Selected Scenario Task IDs =====================
SELECTED_SCENARIO_TASKS = {
    ("retail", 6): list(range(1, 6)) + list(range(7, 10)) + list(range(11, 14)) + list(range(15, 24)),
    ("retail", 10): list(range(1, 5)) + list(range(6, 12)) + list(range(13, 18)) + list(range(19, 24)),
    ("order", 2): [1, 2, 3, 5, 7, 8, 11, 12, 13, 15, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28],
    ("kitchen", 4): [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ("restaurant", 5): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20],
}

TARGET_MODE = "easy"

# ===================== Merge Similar Items Configuration =====================
MERGE_NAME_KEYS = [
    "product_name",
    "ingredient_name",
    "dish_name",
    "set_meal_name",
    "restaurant_name",
    "recipe_name",
    "name",
]

MERGE_SUM_KEYS = {
    "quantity",
}


# ===================== Numeric Normalization for Hashing =====================
def normalize_for_hash(value):
    """Normalize numeric values so that integer-valued floats hash identically to ints."""
    if isinstance(value, dict):
        return {k: normalize_for_hash(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_for_hash(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_for_hash(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    return value


# ===================== Generic Helpers =====================
def try_parse_number(value):
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            v = value.strip()
            if v == "":
                return None
            return float(v)
    except Exception:
        return None
    return None


def canonical_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


# ===================== Fuzzy Match Function =====================
def fuzzy_match_str(query: str, target: str) -> bool:
    """Generic string fuzzy matching: lowercase containment check"""
    if not query or not target:
        return False
    return query.lower() in target.lower() or target.lower() in query.lower()


# ===================== Scenario-based Fuzzy Matching Core Function =====================
def fuzzy_match_field(gt_name: str, inter_name: str, db_instance: Any, scenario: str) -> bool:
    """
    Perform fuzzy matching based on scenario.
    """
    if not db_instance:
        return fuzzy_match_str(inter_name, gt_name)

    if scenario == "kitchen":
        return gt_name.lower().strip() == inter_name.lower().strip()

    match_method = DB_MATCH_METHOD.get(scenario)
    set_meal_method = DB_SET_MEAL_MATCH_METHOD.get(scenario)

    def _collect_matching_names(name: str) -> set:
        all_names = set()

        if match_method and hasattr(db_instance, match_method):
            try:
                match_func = getattr(db_instance, match_method)
                if scenario == "order":
                    for r_name in db_instance.restaurants:
                        matches = match_func(r_name, name)
                        all_names.update(m.name for m in matches)
                else:
                    matches = match_func(name)
                    all_names.update(m.name for m in matches)
            except Exception:
                pass

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
            except Exception:
                pass

        return all_names

    gt_names = _collect_matching_names(gt_name)
    inter_names = _collect_matching_names(inter_name)

    if len(gt_names) > 0 and len(inter_names) > 0:
        return len(gt_names & inter_names) > 0

    return fuzzy_match_str(inter_name, gt_name)


# ===================== Merge Similar Items Before Matching =====================
def get_merge_identity_key(item: dict):
    """
    Return the first available name-like key for grouping similar dict items.
    """
    for key in MERGE_NAME_KEYS:
        if key in item and item[key] is not None:
            return key
    return None


def merge_two_dict_items(base: dict, incoming: dict) -> dict:
    """
    Merge two similar dict items:
    - sum numeric quantity-like fields in MERGE_SUM_KEYS
    - for other fields, keep existing non-empty value; otherwise fill from incoming
    """
    merged = dict(base)

    for k, v in incoming.items():
        if k in MERGE_SUM_KEYS:
            a = try_parse_number(merged.get(k))
            b = try_parse_number(v)
            if a is not None and b is not None:
                summed = a + b
                merged[k] = int(summed) if summed.is_integer() else summed
            elif merged.get(k) is None:
                merged[k] = v
        else:
            existing = merged.get(k)
            if existing is None or (isinstance(existing, str) and existing.strip() == ""):
                merged[k] = v

    return merged


def merge_similar_items_in_list(items, db_instance=None, scenario="retail", current_key=None):
    """
    Merge similar dict items in a list before matching.
    Typical use case:
    [
        {"dish_name": "apple", "quantity": 1},
        {"dish_name": "apple", "quantity": 2}
    ]
    =>
    [
        {"dish_name": "apple", "quantity": 3}
    ]

    注意：
    为保证 list 参数评估时是“无序精准匹配”，这里的“similar”也只按严格相等合并，
    不再对 name 类字段使用模糊匹配合并。
    """
    if not isinstance(items, list):
        return items

    normalized_items = [
        merge_similar_items_in_value(item, db_instance=db_instance, scenario=scenario, current_key=current_key)
        for item in items
    ]

    if not normalized_items:
        return normalized_items

    if not all(isinstance(x, dict) for x in normalized_items):
        return normalized_items

    merged_groups = []

    for item in normalized_items:
        name_key = get_merge_identity_key(item)
        if not name_key:
            merged_groups.append(item)
            continue

        current_name = item.get(name_key)
        merged = False

        for idx, existing in enumerate(merged_groups):
            if not isinstance(existing, dict):
                continue
            existing_name_key = get_merge_identity_key(existing)
            if existing_name_key != name_key:
                continue

            existing_name = existing.get(existing_name_key)
            if isinstance(current_name, str) and isinstance(existing_name, str):
                same = canonical_string(existing_name) == canonical_string(current_name)

                if same:
                    merged_groups[idx] = merge_two_dict_items(existing, item)
                    merged = True
                    break
            else:
                if current_name == existing_name:
                    merged_groups[idx] = merge_two_dict_items(existing, item)
                    merged = True
                    break

        if not merged:
            merged_groups.append(item)

    def sort_key(x):
        if isinstance(x, dict):
            name_key = get_merge_identity_key(x)
            if name_key:
                return canonical_string(x.get(name_key))
        return json.dumps(normalize_for_hash(x), ensure_ascii=False, sort_keys=True, default=str)

    return sorted(merged_groups, key=sort_key)


def merge_similar_items_in_value(value, db_instance=None, scenario="retail", current_key=None):
    """
    Recursively merge similar items in complex parameter values.
    """
    if isinstance(value, dict):
        return {
            k: merge_similar_items_in_value(v, db_instance=db_instance, scenario=scenario, current_key=k)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return merge_similar_items_in_list(value, db_instance=db_instance, scenario=scenario, current_key=current_key)
    if isinstance(value, tuple):
        return [
            merge_similar_items_in_value(v, db_instance=db_instance, scenario=scenario, current_key=current_key)
            for v in value
        ]
    return value


def normalize_call_parameters_before_match(params, db_instance=None, scenario="retail"):
    """
    Normalize parameters before comparison:
    1. recursively process nested values
    2. merge similar items in lists
    """
    if not isinstance(params, dict):
        return params
    return merge_similar_items_in_value(params, db_instance=db_instance, scenario=scenario)


# ===================== Strict / Exact Comparison Helpers =====================
def compare_parameters_recursive_exact(
    gt_val: Any,
    inter_val: Any,
    db_instance: Any = None,
    scenario: str = "retail",
    current_key: str = None
) -> bool:
    """
    Exact recursive comparison.
    For list:
    - unordered exact matching only
    For fuzzy keys:
    - exact means normalized string equality, NOT fuzzy db matching
    """
    if type(gt_val) != type(inter_val):
        try:
            if isinstance(gt_val, (str, int, float)) and isinstance(inter_val, (str, int, float)):
                return float(gt_val) == float(inter_val)
        except (ValueError, TypeError):
            pass
        return False

    if isinstance(gt_val, list):
        gt_val = merge_similar_items_in_list(gt_val, db_instance, scenario, current_key)
        inter_val = merge_similar_items_in_list(inter_val, db_instance, scenario, current_key)

        if len(gt_val) != len(inter_val):
            return False

        inter_used = [False] * len(inter_val)
        for gt_item in gt_val:
            found = False
            for j, inter_item in enumerate(inter_val):
                if inter_used[j]:
                    continue
                if compare_parameters_recursive_exact(gt_item, inter_item, db_instance, scenario, current_key):
                    inter_used[j] = True
                    found = True
                    break
            if not found:
                return False
        return True

    if isinstance(gt_val, dict):
        gt_val = merge_similar_items_in_value(gt_val, db_instance, scenario, current_key)
        inter_val = merge_similar_items_in_value(inter_val, db_instance, scenario, current_key)

        if set(gt_val.keys()) != set(inter_val.keys()):
            return False

        for key, g_val in gt_val.items():
            i_val = inter_val[key]
            if not compare_parameters_recursive_exact(g_val, i_val, db_instance, scenario, key):
                return False
        return True

    if isinstance(gt_val, str):
        if current_key in FUZZY_KEYS.get(scenario, []):
            return canonical_string(gt_val) == canonical_string(inter_val)
        return gt_val == inter_val

    return gt_val == inter_val


# ===================== Recursive Parameter Comparison =====================
def compare_parameters_recursive(
    gt_val: Any,
    inter_val: Any,
    db_instance: Any = None,
    scenario: str = "retail",
    current_key: str = None
) -> bool:
    """
    Recursively compare two parameter values.

    关键修改：
    当参数是 list 时，严格按照“无序精准匹配”进行评估，
    不再使用任何模糊匹配。
    """
    if type(gt_val) != type(inter_val):
        try:
            if isinstance(gt_val, (str, int, float)) and isinstance(inter_val, (str, int, float)):
                return float(gt_val) == float(inter_val)
        except (ValueError, TypeError):
            pass
        return False

    if isinstance(gt_val, list):
        # 严格要求 list 使用无序精准匹配，不进行模糊匹配
        return compare_parameters_recursive_exact(gt_val, inter_val, db_instance, scenario, current_key)

    if isinstance(gt_val, dict):
        gt_val = merge_similar_items_in_value(gt_val, db_instance, scenario, current_key)
        inter_val = merge_similar_items_in_value(inter_val, db_instance, scenario, current_key)

        if set(gt_val.keys()) != set(inter_val.keys()):
            return False

        for key, g_val in gt_val.items():
            i_val = inter_val[key]
            if key in FUZZY_KEYS.get(scenario, []):
                if isinstance(g_val, str):
                    if not fuzzy_match_field(g_val, i_val, db_instance, scenario):
                        return False
                else:
                    if not compare_parameters_recursive(g_val, i_val, db_instance, scenario, key):
                        return False
            else:
                if not compare_parameters_recursive(g_val, i_val, db_instance, scenario, key):
                    return False
        return True

    if isinstance(gt_val, str):
        if current_key in FUZZY_KEYS.get(scenario, []):
            return gt_val.lower().strip() == inter_val.lower().strip()
        return gt_val == inter_val

    return gt_val == inter_val


# ===================== Parameter Comparison Wrapper =====================
def compare_parameters_with_fuzzy_match(
    gt_params: Dict[str, Any],
    interaction_params: Dict[str, Any],
    db_instance: Any = None,
    scenario: str = "retail"
) -> bool:
    gt_params = normalize_call_parameters_before_match(gt_params, db_instance, scenario)
    interaction_params = normalize_call_parameters_before_match(interaction_params, db_instance, scenario)
    return compare_parameters_recursive(gt_params, interaction_params, db_instance, scenario)


# ===================== Database Hash Calculation =====================
def calculate_db_hash(db_instance):
    """Calculate database state hash, supporting all scenario databases"""
    db_data = {}
    if isinstance(db_instance, KitchenDB):
        db_data = {
            'ingredients': {
                k: {
                    'name': v.name, 'quantity': v.quantity, 'category': v.category,
                    'storage_location': v.storage_location, 'expiry_date': v.expiry_date,
                    'nutrition': vars(v.nutrition) if v.nutrition else None
                }
                for k, v in db_instance.ingredients.items()
            },
            'recipes': {
                k: {
                    'name': v.name,
                    'ingredients': [{'ingredient_name': i.ingredient_name, 'quantity': i.quantity} for i in v.ingredients],
                    'allergens': v.allergens,
                    'taste': v.taste,
                    'nutritional_characteristics': v.nutritional_characteristics
                }
                for k, v in db_instance.recipes.items()
            },
            'user_menus': db_instance.user_menus,
            'user_shopping_lists': {
                k: sorted(
                    [{'ingredient_name': item.ingredient_name, 'quantity': item.quantity} for item in v.values()],
                    key=lambda x: x['ingredient_name']
                )
                for k, v in db_instance.user_shopping_lists.items()
            }
        }
    elif isinstance(db_instance, RetailDB):
        db_data = {
            'catalog': {
                k: {
                    'name': v.name, 'category': v.category, 'price': v.price, 'tax_rate': v.tax_rate,
                    'discount': v.discount, 'nutritional_characteristics': v.nutritional_characteristics,
                    'taste': v.taste, 'country_of_origin': v.country_of_origin,
                    'nutrition': vars(v.nutrition) if v.nutrition else None
                }
                for k, v in db_instance.catalog.items()
            },
            'user_carts': {
                k: sorted(
                    [{
                        'product_name': item.product_name, 'quantity': item.quantity,
                        'category': item.category, 'price': item.price, 'tax_rate': item.tax_rate,
                        'discount': item.discount
                    } for item in v.values()],
                    key=lambda x: x['product_name']
                )
                for k, v in db_instance.user_carts.items()
            },
            'user_shopping_lists': db_instance.user_shopping_lists
        }
    elif isinstance(db_instance, RestaurantDB):
        db_data = {
            'catalog': {
                k: {
                    'name': v.name, 'category': v.category, 'price': v.price, 'tax_rate': v.tax_rate,
                    'discount': v.discount, 'nutritional_characteristics': v.nutritional_characteristics,
                    'taste': v.taste, 'allergens': v.allergens,
                    'nutrition': vars(v.nutrition) if v.nutrition else None
                }
                for k, v in db_instance.catalog.items()
            },
            'set_meals': {
                k: {
                    'name': v.name, 'included_dishes': v.included_dishes,
                    'set_meal_price': v.set_meal_price, 'set_meal_discount': v.set_meal_discount
                }
                for k, v in db_instance.set_meals.items()
            },
            'user_orders': {
                k: sorted(
                    [{'dish_name': item.dish_name, 'quantity': item.quantity} for item in v.values()],
                    key=lambda x: x['dish_name']
                )
                for k, v in db_instance.user_orders.items()
            }
        }
    elif isinstance(db_instance, OrderDB):
        db_data = {'restaurants': {}}
        for r_name, store in db_instance.restaurants.items():
            db_data['restaurants'][r_name] = {
                'catalog': {
                    k: {
                        'name': v.name, 'category': v.category, 'price': v.price, 'tax_rate': v.tax_rate,
                        'discount': v.discount, 'nutritional_characteristics': v.nutritional_characteristics,
                        'taste': v.taste, 'allergens': v.allergens,
                        'nutrition': vars(v.nutrition) if v.nutrition else None
                    }
                    for k, v in store['catalog'].items()
                },
                'set_meals': {
                    k: {
                        'name': v.name, 'included_dishes': v.included_dishes,
                        'set_meal_price': v.set_meal_price, 'set_meal_discount': v.set_meal_discount
                    }
                    for k, v in store['set_meals'].items()
                },
                'user_orders': {
                    k: sorted(
                        [{'dish_name': item.dish_name, 'quantity': item.quantity} for item in v.values()],
                        key=lambda x: x['dish_name']
                    )
                    for k, v in store['user_orders'].items()
                }
            }
    elif hasattr(db_instance, 'get_all_data'):
        db_data = db_instance.get_all_data()
    else:
        for attr in dir(db_instance):
            if not attr.startswith('_') and not callable(getattr(db_instance, attr)):
                attr_value = getattr(db_instance, attr)
                try:
                    json.dumps(attr_value, sort_keys=True, default=str)
                    db_data[attr] = attr_value
                except Exception:
                    continue

    json_str = json.dumps(normalize_for_hash(db_data), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


# ===================== Ground Truth Tool Call Simplification =====================
def simplify_tool_calls(db_instance, tool_calls, scenario="retail"):
    """
    Simplify ground truth tool calls, keeping only parameters needed by database methods.
    Also normalize parameters by merging similar items before matching/execution.
    """
    simplified_calls = []
    for tool_call in tool_calls:
        try:
            method_name = tool_call.get("tool_name") or tool_call.get("name")
            params = tool_call.get("parameters", {})

            if hasattr(db_instance, method_name):
                method = getattr(db_instance, method_name)
                sig = inspect.signature(method)
                valid_params = {k: v for k, v in params.items() if k in sig.parameters}
                valid_params = normalize_call_parameters_before_match(valid_params, db_instance, scenario)
                simplified_calls.append({
                    "tool_name": method_name,
                    "parameters": valid_params
                })
            else:
                tool_call_copy = dict(tool_call)
                if "parameters" in tool_call_copy:
                    tool_call_copy["parameters"] = normalize_call_parameters_before_match(
                        tool_call_copy.get("parameters", {}), db_instance, scenario
                    )
                simplified_calls.append(tool_call_copy)
        except Exception:
            simplified_calls.append(tool_call)

    return simplified_calls


# ===================== Tool Execution Function =====================
def execute_tool_chain(db_instance, tool_calls, scenario="retail"):
    """Execute tool call chain with parameter filtering and pre-merge normalization."""
    results = []
    for tool_call in tool_calls:
        try:
            method_name = tool_call.get("tool_name") or tool_call.get("name")
            params = tool_call.get("parameters", {})

            if hasattr(db_instance, method_name):
                method = getattr(db_instance, method_name)
                sig = inspect.signature(method)
                valid_params = {k: v for k, v in params.items() if k in sig.parameters}
                valid_params = normalize_call_parameters_before_match(valid_params, db_instance, scenario)

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


# ===================== Tool Call Comparison =====================
def compare_tool_calls(ground_truth_calls, interaction_calls, db_instance=None, scenario="retail"):
    """
    Compare tool calls.
    Before matching, merge similar items in parameters.
    """
    def extract_call_info(call):
        if isinstance(call, dict):
            return {
                "tool_name": call.get("tool_name") or call.get("name"),
                "parameters": call.get("parameters", {})
            }
        return call

    def filter_params_by_method(tool_name, params, db):
        if db and hasattr(db, tool_name):
            try:
                method = getattr(db, tool_name)
                sig = inspect.signature(method)
                params = {k: v for k, v in params.items() if k in sig.parameters}
            except Exception:
                pass
        return normalize_call_parameters_before_match(params, db, scenario)

    try:
        gt_calls = [extract_call_info(call) for call in ground_truth_calls]
        for call in gt_calls:
            call["parameters"] = normalize_call_parameters_before_match(
                call.get("parameters", {}), db_instance, scenario
            )

        interaction_only_calls = []
        for entry in interaction_calls:
            if isinstance(entry, dict):
                if "calls" in entry and isinstance(entry["calls"], list):
                    for call in entry["calls"]:
                        call_info = extract_call_info(call)
                        call_info["parameters"] = filter_params_by_method(
                            call_info["tool_name"],
                            call_info["parameters"],
                            db_instance
                        )
                        interaction_only_calls.append(call_info)
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
    except Exception:
        return 0, 0, 0


# ===================== Database Initialization =====================
def get_init_db(scenario, scenario_number):
    import io

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
                db.init_from_json(init_data[scenario_number - 1])
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
def evaluate_interaction_success(
    ground_truth_file,
    interaction_log_file,
    scenario="kitchen",
    args=None,
    silent=False,
    num_samples=0,
    allowed_task_ids=None
):
    """
    Evaluate interaction success rate.
    """
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    with open(interaction_log_file, 'r', encoding='utf-8') as f:
        interaction_data = json.load(f)

    if allowed_task_ids is not None:
        allowed_task_ids = set(allowed_task_ids)

        filtered_ground_truth_data = []
        filtered_interaction_data = []

        max_len = min(len(ground_truth_data), len(interaction_data))
        for idx in range(max_len):
            gt_item = ground_truth_data[idx]
            task_id = gt_item.get("task_id", idx + 1)
            if task_id in allowed_task_ids:
                filtered_ground_truth_data.append(gt_item)
                filtered_interaction_data.append(interaction_data[idx])

        ground_truth_data = filtered_ground_truth_data
        interaction_data = filtered_interaction_data

    if num_samples > 0:
        ground_truth_data = ground_truth_data[:num_samples]
        interaction_data = interaction_data[:num_samples]

    results = {
        "total_scenarios": len(ground_truth_data),
        "valid_scenarios": 0,
        "invalid_scenarios": [],
        "tool_based": {"success_count": 0, "partial_matches": [], "success_rate": 0.0},
        "result_based": {"success_count": 0, "success_rate": 0.0},
        "joint_success": {"success_count": 0, "success_rate": 0.0},
        "detailed_results": [],
        "micro_tool_stats": {
            "total_correct_calls": 0, "total_ground_truth_calls": 0,
            "total_interaction_calls": 0, "micro_accuracy": 0.0,
            "task_count": 0
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
        "scenario_stats": {
            "per_scenario": [],
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

    for task_idx in range(len(ground_truth_data)):
        if task_idx >= len(interaction_data):
            if not silent:
                print(f"Warning: interaction log missing scenario {task_idx + 1} data, skipped")
            results["invalid_scenarios"].append({
                "task_id": task_idx + 1,
                "reason": "Missing interaction data"
            })
            continue

        gt_scenario = ground_truth_data[task_idx]
        interaction_scenario = interaction_data[task_idx]
        task_id = gt_scenario.get("task_id", task_idx + 1)

        gt_tool_calls_raw = gt_scenario.get("ground_truth", [])
        interaction_tool_calls = interaction_scenario.get("tool_calls", [])
        if not isinstance(gt_tool_calls_raw, list) or not isinstance(interaction_tool_calls, list):
            results["invalid_scenarios"].append({
                "task_id": task_id,
                "reason": "Data format error (not a list)"
            })
            continue

        valid_interaction_count += 1
        results["valid_scenarios"] += 1

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
            "joint_success": False
        }

        db_instance_for_matching = get_init_db(scenario, args.scenario_number)
        gt_tool_calls = simplify_tool_calls(db_instance_for_matching, gt_tool_calls_raw, scenario=scenario)

        matches, total_gt, total_interactions = compare_tool_calls(
            gt_tool_calls, interaction_tool_calls, db_instance_for_matching, scenario
        )
        detailed_result["tool_based"].update({
            "matches": matches,
            "total_gt_calls": total_gt,
            "total_interaction_calls": total_interactions
        })

        total_correct_calls += matches
        total_gt_calls += total_gt
        total_interaction_calls += total_interactions

        tool_success = False
        if matches == total_gt and total_gt > 0:
            results["tool_based"]["success_count"] += 1
            detailed_result["tool_based"]["success"] = True
            tool_success = True
        elif matches > 0:
            results["tool_based"]["partial_matches"].append({
                "task_id": task_id,
                "matches": matches,
                "total": total_gt
            })

        result_success = False
        try:
            gt_db = get_init_db(scenario, args.scenario_number)
            gt_tool_calls = execute_tool_chain(gt_db, gt_tool_calls, scenario=scenario)
            gt_hash = calculate_db_hash(gt_db)

            interaction_db = get_init_db(scenario, args.scenario_number)
            interaction_only_calls = []
            for entry in interaction_tool_calls:
                if isinstance(entry, dict):
                    if "calls" in entry and isinstance(entry["calls"], list):
                        interaction_only_calls.extend(entry["calls"])
                    elif "call" in entry:
                        interaction_only_calls.append(entry["call"])
            interaction_only_calls = execute_tool_chain(interaction_db, interaction_only_calls, scenario=scenario)
            interaction_hash = calculate_db_hash(interaction_db)

            detailed_result["result_based"].update({
                "gt_hash": gt_hash,
                "interaction_hash": interaction_hash
            })
            if gt_hash == interaction_hash:
                results["result_based"]["success_count"] += 1
                detailed_result["result_based"]["success"] = True
                result_success = True
        except Exception as e:
            if not silent:
                print(f"Scenario {task_id} database operation failed, skipped result evaluation: {e}")

        if tool_success and result_success:
            results["joint_success"]["success_count"] += 1
            detailed_result["joint_success"] = True

        corrected_scores = interaction_scenario.get("corrected_scores", {})
        user_has_issue = False
        if corrected_scores:
            for score_key in ["role_consistency", "instruction_following", "resilience", "contextual_robustness"]:
                if corrected_scores.get(score_key, 1) == 0:
                    user_has_issue = True
                    break
        detailed_result["user_has_issue"] = user_has_issue

        scenario_tokens = interaction_scenario.get("tokens_consumed", 0)
        scenario_tool_calls = len(interaction_tool_calls)

        results["scenario_stats"]["per_scenario"].append({
            "task_id": task_id,
            "tool_calls": scenario_tool_calls,
            "tokens_consumed": scenario_tokens,
            "is_success": tool_success and result_success,
            "user_has_issue": user_has_issue
        })

        if tool_success and result_success:
            results["scenario_stats"]["success_scenarios"]["total_tool_calls"] += scenario_tool_calls
            results["scenario_stats"]["success_scenarios"]["total_tokens"] += scenario_tokens
            results["scenario_stats"]["success_scenarios"]["count"] += 1
        else:
            results["scenario_stats"]["failure_scenarios"]["total_tool_calls"] += scenario_tool_calls
            results["scenario_stats"]["failure_scenarios"]["total_tokens"] += scenario_tokens
            results["scenario_stats"]["failure_scenarios"]["count"] += 1

        results["detailed_results"].append(detailed_result)

    if results["valid_scenarios"] > 0:
        results["tool_based"]["success_rate"] = results["tool_based"]["success_count"] / results["valid_scenarios"]
        results["result_based"]["success_rate"] = results["result_based"]["success_count"] / results["valid_scenarios"]
        results["joint_success"]["success_rate"] = results["joint_success"]["success_count"] / results["valid_scenarios"]

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

    task_count = len(results["detailed_results"])
    results["micro_tool_stats"].update({
        "total_correct_calls": total_correct_calls,
        "total_ground_truth_calls": total_gt_calls,
        "total_interaction_calls": total_interaction_calls,
        "task_count": task_count,
        "micro_accuracy": total_correct_calls / total_gt_calls if total_gt_calls > 0 else 0.0,
        "avg_task_accuracy": (
            sum(
                d["tool_based"]["matches"] / d["tool_based"]["total_gt_calls"]
                for d in results["detailed_results"]
                if d["tool_based"]["total_gt_calls"] > 0
            ) / task_count if task_count > 0 else 0.0
        )
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
        "filtered_avg_task_accuracy": (
            sum(
                d["tool_based"]["matches"] / d["tool_based"]["total_gt_calls"]
                for d in filtered_details
                if d["tool_based"]["total_gt_calls"] > 0
            ) / filtered_valid if filtered_valid > 0 else 0.0
        )
    }

    return results


# ===================== Main Function =====================
def main():
    import re

    parser = argparse.ArgumentParser(description="evaluation script")
    parser.add_argument("--model_name", type=str, default="", help="Model name (subdirectory under results folder); empty means evaluate all models")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples per scenario to test, 0 means test all samples")
    parser.add_argument(
        "--all_tasks",
        action="store_true",
        help="Evaluate every ground-truth task in the selected scenarios instead of SELECTED_SCENARIO_TASKS",
    )
    parser.add_argument(
        "--overall_summary_file",
        type=str,
        default="",
        help="Optional overall summary output path; defaults to ../eval_result/overall_summary.json",
    )
    args = parser.parse_args()

    requested_model_name = str(args.model_name or "").strip()
    num_samples = args.num_samples
    base_results_dir = "../results"
    base_output_dir = "../eval_result"
    os.makedirs(base_output_dir, exist_ok=True)

    if not os.path.exists(base_results_dir):
        print(f"Error: results directory '{base_results_dir}' does not exist")
        return

    selected_targets = set(SELECTED_SCENARIO_TASKS.keys())
    file_pattern = re.compile(r'^(\d{8}_\d{6})_([a-z]+)(\d+)_(easy|hard|static)\.json$')

    def get_ground_truth_task_ids(scenario, scenario_number):
        """Load task IDs in ground-truth order for one selected scenario."""
        ground_truth_file = os.path.join(
            "..", "scenarios", "test_GT", f"{scenario}{scenario_number}.json"
        )
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth_data = json.load(f)
        return [
            item.get("task_id", idx + 1)
            for idx, item in enumerate(ground_truth_data)
        ]

    if args.all_tasks:
        evaluation_task_ids = {
            target: get_ground_truth_task_ids(*target)
            for target in SELECTED_SCENARIO_TASKS
        }
    else:
        evaluation_task_ids = {
            target: list(task_ids)
            for target, task_ids in SELECTED_SCENARIO_TASKS.items()
        }

    def find_timestamped_json_files(results_dir):
        """Find timestamped result files, input: result dir -> output: relative paths"""
        json_files = []
        for root, _, files in os.walk(results_dir):
            for file_name in files:
                if not file_name.endswith(".json"):
                    continue
                if not file_pattern.match(file_name):
                    continue
                rel_path = os.path.relpath(os.path.join(root, file_name), results_dir)
                json_files.append(rel_path)
        return sorted(json_files)

    if requested_model_name:
        requested_model_dir = os.path.join(base_results_dir, requested_model_name)
        if not os.path.isdir(requested_model_dir):
            print(f"Error: requested model directory '{requested_model_dir}' does not exist")
            return
        model_roots = [(requested_model_name, requested_model_dir, os.path.join(base_output_dir, requested_model_name))]
    else:
        model_roots = []
        for model_name in sorted(os.listdir(base_results_dir)):
            results_dir = os.path.join(base_results_dir, model_name)
            if not os.path.isdir(results_dir):
                continue
            if find_timestamped_json_files(results_dir):
                model_roots.append((model_name, results_dir, os.path.join(base_output_dir, model_name)))

    if not model_roots:
        print(f"Error: no model directories with timestamped JSON files found in '{base_results_dir}'")
        return

    print(f"Found {len(model_roots)} model directories under {base_results_dir}")
    print(f"Only evaluating selected scenarios in '{TARGET_MODE}' mode.\n")

    overall_summary = {
        "num_samples": num_samples,
        "mode": TARGET_MODE,
        "task_scope": "all" if args.all_tasks else "selected",
        "selected_scenarios": {
            f"{scenario}{number}": task_ids
            for (scenario, number), task_ids in evaluation_task_ids.items()
        },
        "models": []
    }

    for model_name, results_dir, output_dir in model_roots:
        os.makedirs(output_dir, exist_ok=True)

        json_files = find_timestamped_json_files(results_dir)
        if not json_files:
            print(f"Skipping model '{model_name}': no timestamped JSON files found")
            continue

        print(f"{'=' * 100}")
        print(f"Evaluating model: {model_name}")
        print(f"{'=' * 100}")

        all_results = []

        for json_file in sorted(json_files):
            json_name = os.path.basename(json_file)
            match = file_pattern.match(json_name)
            if not match:
                continue

            timestamp = match.group(1)
            scenario_prefix = match.group(2)
            scenario_number = int(match.group(3))
            user_mode = match.group(4)

            if user_mode != TARGET_MODE:
                continue

            if (scenario_prefix, scenario_number) not in selected_targets:
                continue

            ground_truth_file = f"../scenarios/test_GT/{scenario_prefix}{scenario_number}.json"
            interaction_log_file = os.path.join(results_dir, json_file)

            if not os.path.exists(ground_truth_file):
                print(f"Skipped: ground truth file not found '{ground_truth_file}'")
                continue

            scenario_key = (scenario_prefix, scenario_number)
            scenario_task_ids = evaluation_task_ids[scenario_key]
            allowed_task_ids = None if args.all_tasks else scenario_task_ids
            evaluated_task_ids = (
                scenario_task_ids[:num_samples]
                if num_samples > 0
                else scenario_task_ids
            )

            print(f"Evaluating: {model_name}/{json_file} ...", end=" ")

            try:
                results = evaluate_interaction_success(
                    ground_truth_file,
                    interaction_log_file,
                    scenario=scenario_prefix,
                    args=argparse.Namespace(scenario_number=scenario_number),
                    silent=True,
                    num_samples=num_samples,
                    allowed_task_ids=allowed_task_ids
                )

                output_file = os.path.join(output_dir, json_file.replace('.json', '_eval.json'))
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                fui = results.get("filtered_user_issue", {})
                print(
                    f"done | Valid: {results['valid_scenarios']}/{results['total_scenarios']}, "
                    f"Tool: {results['tool_based']['success_rate']:.2%}, "
                    f"Result: {results['result_based']['success_rate']:.2%}, "
                    f"Joint: {results['joint_success']['success_rate']:.2%}, "
                    f"Filtered Joint: {fui.get('filtered_joint_success_rate', 0):.2%}"
                )

                scenario_stats = results.get("scenario_stats", {})
                success_stats = scenario_stats.get("success_scenarios", {})
                failure_stats = scenario_stats.get("failure_scenarios", {})

                all_results.append({
                    "file": json_file,
                    "timestamp": timestamp,
                    "scenario": scenario_prefix,
                    "scenario_number": scenario_number,
                    "mode": user_mode,
                    "evaluated_task_ids": evaluated_task_ids,
                    "total_scenarios": results["total_scenarios"],
                    "valid_scenarios": results["valid_scenarios"],
                    "tool_based_success_rate": results["tool_based"]["success_rate"],
                    "result_based_success_rate": results["result_based"]["success_rate"],
                    "joint_success_rate": results["joint_success"]["success_rate"],
                    "micro_accuracy": results["micro_tool_stats"]["micro_accuracy"],
                    "avg_task_accuracy": results["micro_tool_stats"]["avg_task_accuracy"],
                    "success_scenario_count": success_stats.get("count", 0),
                    "failure_scenario_count": failure_stats.get("count", 0),
                    "success_avg_tool_calls": success_stats.get("avg_tool_calls", 0),
                    "failure_avg_tool_calls": failure_stats.get("avg_tool_calls", 0),
                    "success_avg_tokens": success_stats.get("avg_tokens", 0),
                    "failure_avg_tokens": failure_stats.get("avg_tokens", 0),
                    "avg_rounds_count": results["performance_metrics"]["avg_rounds_count"],
                    "avg_input_tokens": results["performance_metrics"]["avg_input_tokens"],
                    "avg_output_tokens": results["performance_metrics"]["avg_output_tokens"],
                    "avg_tool_calls_count": results["performance_metrics"]["avg_tool_calls_count"],
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
                print(f"failed: {e}")
                all_results.append({
                    "file": json_file,
                    "timestamp": timestamp,
                    "scenario": scenario_prefix,
                    "scenario_number": scenario_number,
                    "mode": user_mode,
                    "error": str(e)
                })

        valid_results = [r for r in all_results if "error" not in r]

        if valid_results:
            total_valid = sum(r["valid_scenarios"] for r in valid_results)

            avg_tool_success = sum(r["tool_based_success_rate"] * r["valid_scenarios"] for r in valid_results) / total_valid if total_valid > 0 else 0
            avg_result_success = sum(r["result_based_success_rate"] * r["valid_scenarios"] for r in valid_results) / total_valid if total_valid > 0 else 0
            avg_joint_success = sum(r["joint_success_rate"] * r["valid_scenarios"] for r in valid_results) / total_valid if total_valid > 0 else 0
            avg_micro_accuracy = sum(r["micro_accuracy"] * r["valid_scenarios"] for r in valid_results) / total_valid if total_valid > 0 else 0
            avg_task_accuracy = sum(r["avg_task_accuracy"] * r["valid_scenarios"] for r in valid_results) / total_valid if total_valid > 0 else 0

            total_filtered_valid = sum(r.get("filtered_valid_scenarios", 0) for r in valid_results)
            total_user_issue = sum(r.get("user_issue_count", 0) for r in valid_results)

            avg_filtered_tool_success = (
                sum(r["filtered_tool_based_success_rate"] * r["filtered_valid_scenarios"] for r in valid_results) / total_filtered_valid
                if total_filtered_valid > 0 else 0
            )
            avg_filtered_result_success = (
                sum(r["filtered_result_based_success_rate"] * r["filtered_valid_scenarios"] for r in valid_results) / total_filtered_valid
                if total_filtered_valid > 0 else 0
            )
            avg_filtered_joint_success = (
                sum(r["filtered_joint_success_rate"] * r["filtered_valid_scenarios"] for r in valid_results) / total_filtered_valid
                if total_filtered_valid > 0 else 0
            )
            avg_filtered_micro_accuracy = (
                sum(r["filtered_micro_accuracy"] * r["filtered_valid_scenarios"] for r in valid_results) / total_filtered_valid
                if total_filtered_valid > 0 else 0
            )
            avg_filtered_task_accuracy = (
                sum(r["filtered_avg_task_accuracy"] * r["filtered_valid_scenarios"] for r in valid_results) / total_filtered_valid
                if total_filtered_valid > 0 else 0
            )
        else:
            total_valid = 0
            avg_tool_success = avg_result_success = avg_joint_success = 0
            avg_micro_accuracy = avg_task_accuracy = 0
            total_filtered_valid = 0
            total_user_issue = 0
            avg_filtered_tool_success = avg_filtered_result_success = avg_filtered_joint_success = 0
            avg_filtered_micro_accuracy = avg_filtered_task_accuracy = 0

        per_scenario_summary = {}
        for r in valid_results:
            key = f"{r.get('timestamp', '')}_{r['scenario']}{r['scenario_number']}_{r['mode']}"
            per_scenario_summary[key] = {
                "file": r["file"],
                "evaluated_task_ids": r.get("evaluated_task_ids", []),
                "valid_scenarios": r["valid_scenarios"],
                "total_scenarios": r["total_scenarios"],
                "tool_based_success_rate": r["tool_based_success_rate"],
                "result_based_success_rate": r["result_based_success_rate"],
                "joint_success_rate": r["joint_success_rate"],
                "micro_accuracy": r["micro_accuracy"],
                "avg_task_accuracy": r["avg_task_accuracy"],
                "filtered_joint_success_rate": r.get("filtered_joint_success_rate", 0),
                "filtered_micro_accuracy": r.get("filtered_micro_accuracy", 0),
                "filtered_avg_task_accuracy": r.get("filtered_avg_task_accuracy", 0),
            }

        model_summary = {
            "model_name": model_name,
            "mode": TARGET_MODE,
            "total_files": len(all_results),
            "valid_files": len(valid_results),
            "aggregated_metrics": {
                "total_valid_scenarios": total_valid,
                "avg_tool_based_success_rate": avg_tool_success,
                "avg_result_based_success_rate": avg_result_success,
                "avg_joint_success_rate": avg_joint_success,
                "avg_micro_accuracy": avg_micro_accuracy,
                "avg_task_accuracy": avg_task_accuracy,
                "total_user_issue_count": total_user_issue,
                "filtered_valid_scenarios": total_filtered_valid,
                "filtered_tool_based_success_rate": avg_filtered_tool_success,
                "filtered_result_based_success_rate": avg_filtered_result_success,
                "filtered_joint_success_rate": avg_filtered_joint_success,
                "filtered_micro_accuracy": avg_filtered_micro_accuracy,
                "filtered_avg_task_accuracy": avg_filtered_task_accuracy
            },
            "per_scenario_metrics": per_scenario_summary,
            "all_results": all_results
        }

        model_summary_file = os.path.join(output_dir, "summary.json")
        with open(model_summary_file, 'w', encoding='utf-8') as f:
            json.dump(model_summary, f, ensure_ascii=False, indent=2)

        print(f"Saved model summary to: {model_summary_file}\n")

        overall_summary["models"].append(model_summary)

    overall_summary_file = str(args.overall_summary_file or "").strip()
    if not overall_summary_file:
        overall_summary_file = os.path.join(base_output_dir, "overall_summary.json")
    overall_summary_parent = os.path.dirname(overall_summary_file)
    if overall_summary_parent:
        os.makedirs(overall_summary_parent, exist_ok=True)
    with open(overall_summary_file, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(f"{'=' * 100}")
    print(f"All models summary saved to: {overall_summary_file}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
