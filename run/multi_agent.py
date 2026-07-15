import os
import json
import re
import time
import argparse
import sys
from datetime import datetime, timezone, timedelta

# Add the project root directory to Python's module search path
current_file_path = os.path.abspath(__file__)
run_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(run_dir)
sys.path.insert(0, os.path.abspath(project_root))

# 1. Import initialization data
from tools.retail.retail_db import RetailDB
from tools.retail.retail_init import retail_init_data1, retail_init_data2, retail_init_data3, retail_init_data4, retail_init_data5, retail_init_data6, retail_init_data7, retail_init_data8, retail_init_data9, retail_init_data10
from tools.kitchen.kitchen_db import KitchenDB
from tools.kitchen.kitchen_init import kitchen_init_data
from tools.restaurant.restaurant_db import RestaurantDB
from tools.restaurant.restaurant_init import restaurant_init_data, restaurant_init_data5
from tools.order.order_db import OrderDB
from tools.order.order_init import order_init_data
from run.prompts import USER_TEXT_ONLY_PROMPT_EASY
from run.utils import (
    call_llm,
    correct_user_response
)
from run import stage_latency
from run.agent_workers import build_supervisor_prompt, parse_supervisor_action, format_agent_result_for_user
from run.tool_agent import run_tool_agent_plan_and_solve
from run.visual_agent import resolve_expected_visual_names, run_visual_recognition_agent
from config.service_agent_config import get_video_path, SERVICE_MODEL_NAME
from config.visual_agent_config import VISUAL_AGENT_MODEL_NAME
from config.user_agent_config import USER_MODEL_NAME


VISUAL_INPUT_MODE = "image_base64"


def get_media_path_for_visual_model(media_path):
    """Return the configured media path for the current visual video."""
    if not media_path:
        return media_path
    media_filename = os.path.basename(media_path)
    return get_video_path(media_filename)


def _result_task_id(result):
    """Get task id from result, input: result dict -> output: int or None"""
    if not isinstance(result, dict):
        return None
    if result.get("task_id") is not None:
        try:
            return int(result.get("task_id"))
        except (TypeError, ValueError):
            return None
    task_name = str(result.get("task", "")).strip().lower()
    if task_name.startswith("task") and task_name[4:].isdigit():
        return int(task_name[4:])
    return None


def _load_resume_results(output_path):
    """Load checkpoint results, input: output path -> output: result list"""
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print(f"[Warning] Failed to parse existing output file, starting fresh: {output_path}")
        return []


def _sanitize_results_for_save(all_results, keep_tool_debug=False):
    """Strip non-official result fields, input: results/debug flag -> output: sanitized result list"""
    sanitized = []
    for result in all_results:
        if not isinstance(result, dict):
            sanitized.append(result)
            continue
        item = dict(result)
        item.pop("visual_facts", None)
        if not keep_tool_debug:
            item.pop("tool_agent_debug", None)
            item.pop("visual_agent_debug", None)
        sanitized.append(item)
    return sanitized


def _save_resume_results(output_path, all_results, keep_tool_debug=False):
    """Save checkpoint results, input: output path/results -> output: none"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_results_for_save(all_results, keep_tool_debug), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def _compact_parameter_schema(schema):
    """Compress parameter schema, input: JSON schema dict -> output: compact dict"""
    if not isinstance(schema, dict):
        return {}

    compact = {}
    for key in ("type", "enum", "description"):
        if key in schema:
            compact[key] = schema[key]

    properties = schema.get("properties")
    if isinstance(properties, dict) and properties:
        compact["properties"] = {
            name: _compact_parameter_schema(value)
            for name, value in properties.items()
        }

    items = schema.get("items")
    if isinstance(items, dict) and items:
        compact["items"] = _compact_parameter_schema(items)

    required = schema.get("required")
    if isinstance(required, list) and required:
        compact["required"] = required

    return compact


def compact_tool_descriptions(tools_list):
    """Build compact tool descriptions, input: original tool list -> output: compact JSON text"""
    compact_tools = []
    for tool in tools_list:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        compact_tools.append({
            "name": function.get("tool_name") or function.get("name"),
            "description": function.get("description", ""),
            "parameters": {
                name: _compact_parameter_schema(schema)
                for name, schema in properties.items()
            },
            "required": parameters.get("required", []) if isinstance(parameters, dict) else [],
        })
    return json.dumps(compact_tools, ensure_ascii=False, separators=(",", ":"))


def run_simulation(input_path, tool_info_path, output_path, args=None, service_model_name="qwen3-vl-225b"):
    """
    Interactive Mode: Multi-round conversation (Easy mode only)
    """
    use_vision = False

    with open(tool_info_path, 'r', encoding='utf-8') as f:
        tools_list = json.load(f)
        tool_descriptions = compact_tool_descriptions(tools_list)

    if not os.path.exists(input_path):
        print(f"Can't find the file {input_path}.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)

    task_id_offset = 1
    if "," in args.num_tasks:
        start_idx, end_idx = [int(x.strip()) for x in args.num_tasks.split(",", 1)]
        scenarios = scenarios[start_idx - 1:end_idx]
        task_id_offset = start_idx
    elif int(args.num_tasks) > 0:
        scenarios = scenarios[:int(args.num_tasks)]

    all_results = _load_resume_results(output_path)
    completed_task_ids = {task_id for task_id in (_result_task_id(result) for result in all_results) if task_id is not None}
    if all_results:
        print(f"[Resume] Loaded {len(all_results)} existing results from: {output_path}")

    for idx, sc in enumerate(scenarios, start=task_id_offset):
        task_id = idx
        if task_id in completed_task_ids:
            print(f"[Resume] Skip completed task {task_id}")
            continue
        stage_latency.set_context(
            scenario=f"{args.scenario}{args.scenario_number}",
            task_id=task_id,
            turn=None
        )

        print(f"\n{'='*20} Scenario {args.scenario}{args.scenario_number}: {task_id} {'='*20} ")
        if args.scenario == "retail":
            db = RetailDB()
            if args.scenario_number == 1:
                db.init_from_json(retail_init_data1)
            elif args.scenario_number == 2:
                db.init_from_json(retail_init_data2)
            elif args.scenario_number == 3:
                db.init_from_json(retail_init_data3)
            elif args.scenario_number == 4:
                db.init_from_json(retail_init_data4)
            elif args.scenario_number == 5:
                db.init_from_json(retail_init_data5)
            elif args.scenario_number == 6:
                db.init_from_json(retail_init_data6)
            elif args.scenario_number == 7:
                db.init_from_json(retail_init_data7)
            elif args.scenario_number == 8:
                db.init_from_json(retail_init_data8)
            elif args.scenario_number == 9:
                db.init_from_json(retail_init_data9)
            elif args.scenario_number == 10:
                db.init_from_json(retail_init_data10)
        elif args.scenario == "kitchen":
            db = KitchenDB()
            db.init_from_json(kitchen_init_data)
        elif args.scenario == "restaurant":
            db = RestaurantDB()
            if args.scenario_number == 5:
                db.init_from_json(restaurant_init_data5)
            else:
                db.init_from_json(restaurant_init_data)
        elif args.scenario == "order":
            db = OrderDB()
            db.init_from_json(order_init_data)

        user_instruction = sc.get("Instruction", "")
        image_path = sc.get("image_path", None)
        image_path = get_media_path_for_visual_model(image_path)
        if hasattr(db, "set_vision_context"):
            db.set_vision_context(
                media_path=image_path,
                service_model_name=VISUAL_AGENT_MODEL_NAME,
                video_model=VISUAL_INPUT_MODE,
                box_enabled=getattr(args, "box_enabled", True)
            )
        else:
            db.vision_media_path = image_path
            db.vision_service_model_name = VISUAL_AGENT_MODEL_NAME
            db.vision_video_model = VISUAL_INPUT_MODE
            db.vision_box_enabled = getattr(args, "box_enabled", True)
        db.vision_scenario = f"{args.scenario}{args.scenario_number}"
        db.vision_task_id = task_id
        db.vision_run_id = getattr(args, "run_id", "")
        db.vision_request_seq = 0
        db.vision_box_enabled = getattr(args, "box_enabled", True)
        image_description = sc.get("image_description", "")

        start_time = time.time()

        history_log = {
            "task_id": task_id,
            "mode": "text",
            "instruction": user_instruction,
            "image_description": image_description,
            "dialogue": [],
            "visual_facts": [],
            "tool_calls": [],
            "rounds_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_calls_count": 0,
            "user_response_time_seconds": 0.0,
            "agent_response_time_seconds": 0.0,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        }

        user_agent_sys_prompt = USER_TEXT_ONLY_PROMPT_EASY.format(
            user_instruction=user_instruction,
            image_description=image_description,
            original_user_response="",
            evaluation_feedback="",
            history_summary="",
            service_agent_response="Dear customer, how can I help you?"
        )

        user_messages = [
            {"role": "system", "content": user_agent_sys_prompt},
            {"role": "user", "content": "You are a customer in the environment shown in the video, and you need to complete the instructions in **Task**. I am your AI customer service representative; please interact with me in the first person. Let's begin the conversation.\nDear customer, how can I help you?"}
        ]

        service_agent_sys_prompt = build_supervisor_prompt(tool_descriptions)
        service_history = []

        if getattr(args, "test_visual", False):
            def save_visual_test_result(visual_task, visual_result):
                """Save compact visual result, input: visual task/result -> output: none"""
                expected_names = resolve_expected_visual_names(db, sc)
                correct_name = expected_names[0] if len(expected_names) == 1 else expected_names
                result_text = visual_result.get("result") or visual_result.get("message", "")
                print(f"Visual Agent Result: {result_text}")
                print(f"Correct Visual Name: {correct_name}")
                all_results.append({
                    "task": f"task{task_id}",
                    "result": result_text,
                    "raw_response": visual_result.get("raw_response") or visual_result.get("result", ""),
                    "supervisor_task": visual_task,
                    "correct_name": correct_name
                })
                completed_task_ids.add(task_id)
                _save_resume_results(output_path, all_results, getattr(args, "tool_debug", False))

            visual_recorded = False
            last_agent_response_for_check = "Dear customer, how can I help you?"
            for turn in range(10):
                stage_latency.set_context(turn=turn)
                user_start_time = time.time()
                user_profile = stage_latency.start("user_agent", model=USER_MODEL_NAME)
                try:
                    user_reply, user_input_tok, user_output_tok = call_llm(
                        user_messages,
                        agent_type="user",
                        service_model_name=args.service_model_name
                    )
                except Exception:
                    stage_latency.end(user_profile, status="error")
                    raise
                else:
                    stage_latency.end(user_profile, user_input_tok, user_output_tok)
                user_gen_time = time.time() - user_start_time
                print(f"[Time] Visual test user generation (Task {task_id}, Turn {turn}): {user_gen_time:.3f} seconds")
                print(f"Visual Test User Response: {user_reply}")

                evaluation_info = None
                check_start_time = time.time()
                if args.multi_agent_user:
                    original_user_reply = user_reply
                    user_reply, evaluation_info = correct_user_response(
                        user_response=original_user_reply,
                        user_instruction=user_instruction,
                        dialogue=history_log["dialogue"],
                        last_agent_response=last_agent_response_for_check
                    )
                    check_time = time.time() - check_start_time
                    print(f"[Time] Visual test check phase (Task {task_id}, Turn {turn}): {check_time:.3f} seconds")
                    if evaluation_info:
                        print(f"\n[User Response Correction]")
                        print(f"  Correction Applied: {evaluation_info.get('correction_applied', False)}")
                        if evaluation_info.get("correction_reason"):
                            print(f"  Correction Reason: {evaluation_info.get('correction_reason')}")
                    if user_reply != original_user_reply:
                        print(f"Visual Test User Response Corrected: {user_reply}")

                history_log["dialogue"].append({"role": "user", "turn": turn, "content": user_reply})
                if "STOP" in user_reply:
                    print("Stop signal detected before visual recognition")
                    break

                service_history.append({"role": "user", "content": user_reply})
                user_messages.append({"role": "assistant", "content": user_reply})

                supervisor_messages = [{"role": "system", "content": service_agent_sys_prompt}] + service_history
                if args.service_model_name == "manual":
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Manual Supervisor Agent Turn ---")
                    print("Latest User Input:")
                    print(user_reply)
                    print("Enter supervisor routing JSON. Type 'END' on a new line to finish:")
                    ml_input = []
                    while True:
                        try:
                            line = input()
                            if line.strip() == "END":
                                break
                            ml_input.append(line)
                        except EOFError:
                            break
                    supervisor_reply = "\n".join(ml_input)
                else:
                    supervisor_profile = stage_latency.start("supervisor", model=args.service_model_name)
                    try:
                        supervisor_reply, supervisor_input_tok, supervisor_output_tok = call_llm(
                            supervisor_messages,
                            agent_type="service",
                            service_model_name=args.service_model_name
                        )
                    except Exception:
                        stage_latency.end(supervisor_profile, status="error")
                        raise
                    else:
                        stage_latency.end(supervisor_profile, supervisor_input_tok, supervisor_output_tok)
                print(f"Supervisor Agent: {supervisor_reply}")

                supervisor_action = parse_supervisor_action(supervisor_reply, user_reply)
                agent_name = supervisor_action.get("agent_name")
                supervisor_task = supervisor_action.get("task") or user_reply

                if agent_name == "visual_agent":
                    visual_result, _, _ = run_visual_recognition_agent(
                        db,
                        supervisor_task,
                        visual_facts=history_log.get("visual_facts", []),
                    )
                    save_visual_test_result(supervisor_task, visual_result)
                    visual_recorded = True
                    break

                if agent_name == "ask_user":
                    agent_final_reply = supervisor_task
                else:
                    tool_agent_res = run_tool_agent_plan_and_solve(
                        db_instance=db,
                        tool_descriptions=tool_descriptions,
                        task=supervisor_task,
                        user_request=user_reply,
                        dialogue_history=history_log["dialogue"],
                        visual_facts=history_log.get("visual_facts", [])
                    )
                    if tool_agent_res.get("needs_visual_recognition"):
                        visual_requests = tool_agent_res.get("visual_recognition_requests") or []
                        visual_request = next((item for item in visual_requests if isinstance(item, dict)), None)
                        if visual_request:
                            visual_task = str(visual_request.get("visual_task") or supervisor_task).strip()
                            expected_key = visual_request.get("expected_key")
                            visual_result, _, _ = run_visual_recognition_agent(
                                db,
                                visual_task,
                                expected_key=expected_key,
                                visual_facts=history_log.get("visual_facts", []),
                            )
                            save_visual_test_result(visual_task, visual_result)
                            visual_recorded = True
                            break
                        agent_final_reply = "I cannot reliably identify the item from the visual context. Please reply exactly STOP."
                    else:
                        combined_result = tool_agent_res.get("combined_result") or "No tool calls executed."
                        agent_final_reply = format_agent_result_for_user(combined_result)
                        print(f"Tool Agent Raw Result: {combined_result}")
                        print(f"Service Reply: {agent_final_reply}")

                history_log["dialogue"].append({"role": "agent", "turn": turn, "content": agent_final_reply})
                service_history.append({"role": "assistant", "content": agent_final_reply})
                last_agent_response_for_check = agent_final_reply
                user_agent_sys_prompt = USER_TEXT_ONLY_PROMPT_EASY.format(
                    user_instruction=user_instruction,
                    image_description=image_description,
                    original_user_response="",
                    evaluation_feedback="",
                    history_summary="",
                    service_agent_response=last_agent_response_for_check
                )
                user_messages[0]["content"] = user_agent_sys_prompt
                user_messages.append({"role": "user", "content": last_agent_response_for_check})

            if not visual_recorded:
                print(f"[Warning] No visual recognition triggered for task {task_id}; no visual result saved.")
            continue

        max_turns = 10
        rounds_count = 0
        input_tokens_total = 0
        output_tokens_total = 0
        tool_calls_count = 0

        last_agent_response_for_check = "Dear customer, how can I help you?"

        for turn in range(max_turns):
            stage_latency.set_context(turn=turn)
            user_start_time = time.time()
            user_profile = stage_latency.start("user_agent", model=USER_MODEL_NAME)
            try:
                user_reply, user_input_tok, user_output_tok = call_llm(user_messages, agent_type="user", service_model_name=args.service_model_name)
            except Exception:
                stage_latency.end(user_profile, status="error")
                raise
            else:
                stage_latency.end(user_profile, user_input_tok, user_output_tok)
            user_gen_time = time.time() - user_start_time
            print(f"[Time] User response generation (Turn {turn}): {user_gen_time:.3f} seconds")
            history_log["user_response_time_seconds"] += user_gen_time

            evaluation_info = None
            check_start_time = time.time()
            if args.multi_agent_user:
                original_user_reply = user_reply
                user_reply, evaluation_info = correct_user_response(
                    user_response=original_user_reply,
                    user_instruction=user_instruction,
                    dialogue=history_log["dialogue"],
                    last_agent_response=last_agent_response_for_check
                )

                if evaluation_info:
                    print(f"\n[User Response Correction]")
                    print(f"  Correction Applied: {evaluation_info.get('correction_applied', False)}")
                    if evaluation_info.get("correction_reason"):
                        print(f"  Correction Reason: {evaluation_info.get('correction_reason')}")
                if user_reply != original_user_reply:
                    print(f"User Response Corrected: {user_reply}")

            check_time = time.time() - check_start_time
            if args.multi_agent_user:
                print(f"[Time] Check phase (Turn {turn}): {check_time:.3f} seconds")
                history_log["user_response_time_seconds"] += check_time

            print(f"Final User Response: {user_reply}")

            log_entry = {"role": "user", "turn": turn, "content": user_reply}
            if evaluation_info:
                log_entry["evaluation"] = {
                    "correction_applied": bool(evaluation_info.get("correction_applied", False)),
                    "correction_reason": str(evaluation_info.get("correction_reason", "")),
                    "raw_response": str(evaluation_info.get("raw_response", ""))
                }

            history_log["dialogue"].append(log_entry)

            if "STOP" in user_reply:
                print("Stop signal detected")
                break

            service_history.append({"role": "user", "content": user_reply})
            user_messages.append({"role": "assistant", "content": user_reply})

            current_user_reply_for_task = user_reply
            current_service_history = [msg for msg in service_history]

            def process_agent_task():
                agent_start = time.time()
                inner_input_tokens = 0
                inner_output_tokens = 0
                inner_calls = 0
                inner_rounds = 0
                agent_final_reply = ""
                local_tool_logs = []
                local_dialogue_logs = []
                local_service_history = [msg for msg in current_service_history]
                total_tool_calls_so_far = tool_calls_count

                def build_target_description(task_text):
                    """Extract target description, input: visual task text -> output: object description"""
                    text = str(task_text or "").strip()
                    text = re.sub(r"(?i)^please identify the visual target\.\s*task:\s*", "", text).strip()
                    text = re.sub(r"(?i)^please\s+", "", text).strip()
                    text = re.sub(
                        r"(?i)^(identify|find|locate|check|analyze|evaluate|confirm|verify)\s+"
                        r"(the\s+)?(product\s+names?|names?|visual\s+target)\s+(of|for)\s+",
                        "",
                        text,
                    ).strip()
                    text = re.sub(r"(?i)^identify\s+", "", text).strip()
                    return text.rstrip(".") or str(task_text or "").strip()

                def collect_visual_facts(visual_result):
                    """Collect visual facts, input: visual result -> output: visual facts list"""
                    name = visual_result.get("name")
                    if not name:
                        return []
                    if isinstance(name, list) and not name:
                        return []
                    fact = {
                        "task": visual_result.get("task", ""),
                        "target_description": visual_result.get("target_description") or build_target_description(visual_result.get("task", "")),
                        "name": name,
                        "is_latest_recognition": True,
                    }
                    if visual_result.get("key"):
                        fact["key"] = visual_result["key"]
                    return [fact]

                def merge_visual_facts(existing_facts, new_facts):
                    """Merge visual facts, input: existing/new lists -> output: updated fact list"""
                    if isinstance(existing_facts, dict):
                        existing_facts = [existing_facts]
                    if isinstance(new_facts, dict):
                        new_facts = [new_facts]
                    new_facts = [fact for fact in new_facts or [] if isinstance(fact, dict)]
                    if not new_facts:
                        return [dict(fact) for fact in existing_facts or [] if isinstance(fact, dict)]
                    merged = []
                    for fact in existing_facts or []:
                        if isinstance(fact, dict):
                            old_fact = dict(fact)
                            old_fact["is_latest_recognition"] = False
                            merged.append(old_fact)
                    for fact in new_facts:
                        new_fact = dict(fact)
                        new_fact["is_latest_recognition"] = True
                        merged.append(new_fact)
                    return merged

                def save_visual_facts(new_facts):
                    """Persist visual facts, input: new facts -> output: session facts"""
                    if not new_facts:
                        return merge_visual_facts(history_log.get("visual_facts", []), [])
                    history_log["visual_facts"] = merge_visual_facts(
                        history_log.get("visual_facts", []),
                        new_facts
                    )
                    return history_log["visual_facts"]

                def run_tool_task(tool_task, visual_facts, visual_depth=0):
                    """Run tool agent, input: task/visual facts -> output: final reply text"""
                    nonlocal inner_input_tokens, inner_output_tokens, inner_calls
                    tool_agent_res = run_tool_agent_plan_and_solve(
                        db_instance=db,
                        tool_descriptions=tool_descriptions,
                        task=tool_task,
                        user_request=current_user_reply_for_task,
                        dialogue_history=history_log["dialogue"],
                        visual_facts=visual_facts
                    )
                    raw_tool_agent_response = tool_agent_res.get("raw_response", "")
                    if raw_tool_agent_response and getattr(args, "tool_debug", False):
                        try:
                            tool_agent_raw_response = json.loads(raw_tool_agent_response)
                        except json.JSONDecodeError:
                            tool_agent_raw_response = {"raw_response": raw_tool_agent_response}

                        history_log.setdefault("tool_agent_debug", []).append({
                            "turn": turn,
                            "planner_input": {
                                "supervisor_task": tool_task,
                                "visual_facts": visual_facts,
                                "dialogue_history": list(history_log["dialogue"]),
                                "tool_descriptions": tool_descriptions,
                            },
                            "tool_agent_output": {
                                "plan": tool_agent_res.get("plan"),
                                "raw_response": tool_agent_raw_response,
                                "raw_combined_result": tool_agent_res.get("raw_combined_result"),
                                "combined_result": tool_agent_res.get("combined_result"),
                            },
                        })

                    inner_input_tokens += tool_agent_res["input_tokens"]
                    inner_output_tokens += tool_agent_res["output_tokens"]
                    if tool_agent_res.get("needs_visual_recognition"):
                        if visual_depth >= 3:
                            return "I cannot reliably identify the item from the visual context. Please reply exactly STOP."

                        visual_requests = tool_agent_res.get("visual_recognition_requests") or []
                        if not visual_requests:
                            return "I cannot reliably identify the item from the visual context. Please reply exactly STOP."

                        merged_visual_facts = merge_visual_facts(visual_facts, [])
                        for visual_request in visual_requests:
                            if not isinstance(visual_request, dict):
                                continue
                            visual_task = str(visual_request.get("visual_task") or tool_task).strip()
                            expected_key = visual_request.get("expected_key")
                            visual_result, visual_input_tok, visual_output_tok = run_visual_recognition_agent(
                                db,
                                visual_task,
                                expected_key=expected_key,
                                visual_facts=merged_visual_facts,
                            )
                            inner_input_tokens += visual_input_tok
                            inner_output_tokens += visual_output_tok
                            print(f"Visual Agent Result: {visual_result}")
                            if getattr(args, "tool_debug", False):
                                history_log.setdefault("visual_agent_debug", []).append({
                                    "turn": turn,
                                    "source": "tool_planner_visual_recognition_request",
                                    "request": visual_request,
                                    "result": visual_result,
                                })
                            if visual_result.get("status") != "success":
                                return "I cannot reliably identify the item from the visual context. Please reply exactly STOP."
                            new_visual_facts = collect_visual_facts(visual_result)
                            merged_visual_facts = merge_visual_facts(merged_visual_facts, new_visual_facts)
                            save_visual_facts(new_visual_facts)

                        return run_tool_task(tool_task, merged_visual_facts, visual_depth + 1)

                    tool_calls = tool_agent_res["tool_calls"]
                    tool_results = tool_agent_res["tool_results"]
                    inner_calls += len(tool_calls)

                    if tool_calls:
                        local_tool_logs.append({
                            "turn": turn,
                            "calls": tool_calls,
                            "results": tool_results
                        })

                    combined_result = tool_agent_res["combined_result"] or "No tool calls executed."
                    user_visible_result = format_agent_result_for_user(combined_result)
                    print(f"Tool Agent Raw Result: {combined_result}")
                    print(f"Service Reply: {user_visible_result}")
                    if total_tool_calls_so_far + inner_calls > 200:
                        print(f"Tool calls count ({total_tool_calls_so_far + inner_calls}) exceeded 200, stopping interaction.")
                        return "[Interaction stopped: tool calls exceeded 200]"
                    return user_visible_result

                current_service_msgs = [{"role": "system", "content": service_agent_sys_prompt}] + current_service_history

                if args.service_model_name == "manual":
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Manual Supervisor Agent Turn ---")
                    print("Latest User Input:")
                    print(current_user_reply_for_task)
                    print("Enter supervisor routing JSON. Type 'END' on a new line to finish:")
                    ml_input = []
                    while True:
                        try:
                            line = input()
                            if line.strip() == "END":
                                break
                            ml_input.append(line)
                        except EOFError:
                            break
                    agent_reply = "\n".join(ml_input)
                    agent_input_tokens = 0
                    agent_output_tokens = 0
                else:
                    supervisor_profile = stage_latency.start("supervisor", model=args.service_model_name)
                    try:
                        agent_reply, agent_input_tokens, agent_output_tokens = call_llm(
                            current_service_msgs,
                            agent_type="service",
                            service_model_name=args.service_model_name
                        )
                    except Exception:
                        stage_latency.end(supervisor_profile, status="error")
                        raise
                    else:
                        stage_latency.end(supervisor_profile, agent_input_tokens, agent_output_tokens)
                    inner_input_tokens += agent_input_tokens
                    inner_output_tokens += agent_output_tokens
                print(f"Supervisor Agent: {agent_reply}")

                action = parse_supervisor_action(agent_reply, current_user_reply_for_task)
                agent_name = action.get("agent_name")
                agent_task = action.get("task") or current_user_reply_for_task

                if agent_name == "visual_agent":
                    visual_result, visual_input_tok, visual_output_tok = run_visual_recognition_agent(
                        db,
                        agent_task,
                        visual_facts=history_log.get("visual_facts", []),
                    )
                    inner_input_tokens += visual_input_tok
                    inner_output_tokens += visual_output_tok
                    print(f"Visual Agent Result: {visual_result}")
                    if visual_result.get("status") == "success":
                        agent_final_reply = run_tool_task(
                            current_user_reply_for_task,
                            save_visual_facts(collect_visual_facts(visual_result))
                        )
                    else:
                        agent_final_reply = "I cannot reliably identify the item from the visual context. Please reply exactly STOP."
                elif agent_name == "ask_user":
                    agent_final_reply = agent_task
                else:
                    agent_final_reply = run_tool_task(agent_task, save_visual_facts([]))

                inner_rounds += 1
                local_dialogue_logs.append({"role": "agent", "turn": turn, "content": agent_final_reply})
                local_service_history.append({"role": "assistant", "content": agent_final_reply})
                agent_time = time.time() - agent_start
                print(f"[Time] Agent response generation (Turn {turn}): {agent_time:.3f} seconds")
                return {
                    "reply": agent_final_reply,
                    "input_tokens": inner_input_tokens,
                    "output_tokens": inner_output_tokens,
                    "calls": inner_calls,
                    "rounds": inner_rounds,
                    "tool_logs": local_tool_logs,
                    "dialogue_logs": local_dialogue_logs,
                    "time": agent_time,
                    "updated_history": local_service_history
                }

            agent_res = process_agent_task()

            input_tokens_total += agent_res["input_tokens"]
            output_tokens_total += agent_res["output_tokens"]
            tool_calls_count += agent_res["calls"]
            rounds_count += agent_res["rounds"]
            history_log["agent_response_time_seconds"] += agent_res["time"]
            history_log["tool_calls"].extend(agent_res["tool_logs"])
            history_log["dialogue"].extend(agent_res["dialogue_logs"])
            service_history = agent_res["updated_history"]

            last_agent_response_for_check = agent_res["reply"]

            user_agent_sys_prompt = USER_TEXT_ONLY_PROMPT_EASY.format(
                user_instruction=user_instruction,
                image_description=image_description,
                original_user_response="",
                evaluation_feedback="",
                history_summary="",
                service_agent_response=last_agent_response_for_check
            )

            user_messages[0]["content"] = user_agent_sys_prompt
            user_messages.append({"role": "user", "content": last_agent_response_for_check})

        history_log["rounds_count"] = rounds_count
        history_log["input_tokens"] = input_tokens_total
        history_log["output_tokens"] = output_tokens_total
        history_log["tool_calls_count"] = tool_calls_count

        end_time = time.time()
        execution_time = round(end_time - start_time, 3)
        history_log["execution_time_seconds"] = execution_time
        stage_latency.record_task_summary(history_log)
        all_results.append(history_log)
        completed_task_ids.add(task_id)
        _save_resume_results(output_path, all_results, getattr(args, "tool_debug", False))

    _save_resume_results(output_path, all_results, getattr(args, "tool_debug", False))
    print(f"\nCompleted! Results saved to: {output_path}")
    print(f"Statistics Summary: ")
    for idx, result in enumerate(all_results):
        if getattr(args, "test_visual", False):
            print(f"  {result.get('task', f'task{idx + 1}')}: result={result.get('result', '')}, correct_name={result.get('correct_name', '')}")
        else:
            print(f"  Task {idx+1}: {result['rounds_count']} dialogue rounds, {result['input_tokens']} input tokens, {result['output_tokens']} output tokens, {result['tool_calls_count']} tool calls, {result['execution_time_seconds']} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dialogue simulation in easy mode")
    parser.add_argument(
        "--service_model_name",
        default=SERVICE_MODEL_NAME,
        help="Tested agent model name (default: configured in service_agent_config.py)"
    )

    parser.add_argument(
        "--scenario",
        choices=["retail", "kitchen", "restaurant", "order"],
        default="retail",
        help="Task scenario"
    )

    parser.add_argument(
        "--scenario_number",
        type=int,
        default=1,
        help="Scenario number"
    )

    parser.add_argument(
        "--multi_agent_user",
        action="store_true",
        help="When True, use the user response corrector before sending each simulated user reply"
    )

    parser.add_argument(
        "--num_tasks",
        default="0",
        help="Number of tasks to test. 0 means all, N means first N tasks, and start,end means an inclusive task range."
    )

    parser.add_argument(
        "--test_visual",
        action="store_true",
        help="Only test the visual agent and save compact visual recognition results under processed/visual_test."
    )

    parser.add_argument(
        "--tool_debug",
        action="store_true",
        help="Keep internal tool/visual agent debug fields in result JSON."
    )

    parser.add_argument(
        "--stage_latency",
        default="true",
        choices=["true", "false"],
        help="Enable per-stage latency/token profiling. Default: true."
    )

    parser.add_argument(
        "--box",
        default="true",
        choices=["true", "false"],
        help="Enable visual boxed product grounding before visual recognition. Default: true."
    )

    args = parser.parse_args()

    beijing_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    args.run_id = beijing_time
    args.box_enabled = stage_latency.parse_bool(args.box, default=True)

    INPUT_JSON = f"./scenarios/final/{args.scenario}{args.scenario_number}.json"
    TOOL_INFO_JSON = f"./tools/{args.scenario}/{args.scenario}_tools.json"
    scenario_name = f"{args.scenario}{args.scenario_number}"
    if args.test_visual:
        output_model_name = VISUAL_AGENT_MODEL_NAME
        OUTPUT_JSON = f"./processed/visual_test/{output_model_name}/{VISUAL_INPUT_MODE}/{scenario_name}/{beijing_time}_{scenario_name}_easy.json"
    else:
        output_model_name = args.service_model_name
        OUTPUT_JSON = f"./results/{output_model_name}/{VISUAL_INPUT_MODE}/{scenario_name}/{beijing_time}_{scenario_name}_easy.json"

    profile_root = stage_latency.DEFAULT_PROFILE_ROOT / output_model_name / VISUAL_INPUT_MODE / scenario_name
    profile_name = os.path.splitext(os.path.basename(OUTPUT_JSON))[0]
    stage_latency.configure(
        enabled=stage_latency.parse_bool(args.stage_latency, default=True),
        run_id=args.run_id,
        profile_root=str(profile_root),
        profile_name=profile_name,
    )
    if not os.path.exists(os.path.dirname(OUTPUT_JSON)):
        os.makedirs(os.path.dirname(OUTPUT_JSON))

    # # ------------------------------------------------------------
    # # Debug trace: record all user/service LLM inputs and outputs.
    # # ------------------------------------------------------------
    # trace_dir = os.path.join(project_root, "processed", "test_process")
    # os.makedirs(trace_dir, exist_ok=True)
    # trace_model_name = args.service_model_name.replace("/", "_")
    # trace_path = os.path.join(trace_dir, f"{beijing_time}_{trace_model_name}_jointsucess.md")
    # trace_counter = {"n": 0}
    # real_call_llm = call_llm

    # with open(trace_path, "w", encoding="utf-8") as f:
    #     f.write("# Multi-Agent Test Process\n\n")
    #     f.write("Command args:\n\n```json\n")
    #     f.write(json.dumps(vars(args), ensure_ascii=False, indent=2))
    #     f.write("\n```\n\n")
    #     f.write(f"Output JSON: `{OUTPUT_JSON}`\n\n")

    # def traced_call_llm(messages, agent_type="service", service_model_name=None):
    #     trace_counter["n"] += 1
    #     call_id = trace_counter["n"]
    #     start = time.time()
    #     reply, input_tok, output_tok = real_call_llm(
    #         messages,
    #         agent_type=agent_type,
    #         service_model_name=service_model_name
    #     )
    #     elapsed = time.time() - start

    #     with open(trace_path, "a", encoding="utf-8") as f:
    #         f.write(f"## Call {call_id}: {agent_type}\n\n")
    #         f.write(f"- service_model_name: `{service_model_name}`\n")
    #         f.write(f"- elapsed_seconds: `{elapsed:.3f}`\n")
    #         f.write(f"- input_tokens: `{input_tok}`\n")
    #         f.write(f"- output_tokens: `{output_tok}`\n\n")
    #         f.write("### Input Messages\n\n```json\n")
    #         f.write(json.dumps(messages, ensure_ascii=False, indent=2))
    #         f.write("\n```\n\n")
    #         f.write("### Output\n\n```text\n")
    #         f.write(str(reply))
    #         f.write("\n```\n\n")

    #     return reply, input_tok, output_tok

    # call_llm = traced_call_llm

    # import run.utils as run_utils
    # run_utils.call_llm = traced_call_llm

    try:
        run_simulation(INPUT_JSON, TOOL_INFO_JSON, OUTPUT_JSON, args=args, service_model_name=args.service_model_name)
    finally:
        stage_latency.summarize()
