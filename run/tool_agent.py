"""
Tool agent that performs plan-and-solve before executing database tools.
"""
import json
import re
from typing import Any, Dict, List, Optional

from config.tool_agent_config import (
    TOOL_EXECUTOR_MODEL_NAME,
    TOOL_PLANNER_MODEL_NAME,
    TOOL_REPORTER_MODEL_NAME,
    call_tool_executor_model,
    call_tool_planner_model,
    call_tool_reporter_model,
)
from run.prompts import (
    TOOL_AGENT_EXECUTOR_PROMPT,
    TOOL_AGENT_PLANNER_PROMPT,
    TOOL_AGENT_REPORTER_PROMPT,
)
from run import stage_latency
from run.utils import execute_tool
from run.visual_agent import get_visual_expected_keys


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from text, input: model text -> output: dict or None"""
    text = str(text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _format_dialogue_history(history: List[Dict[str, Any]]) -> str:
    """Convert dialogue history to compact text, input: history list -> output: text"""
    lines = []
    for item in history:
        lines.append(f"{item.get('role', 'unknown')}: {item.get('content', '')}")
    return "\n".join(lines) if lines else "None"


def _load_tool_descriptions(tool_descriptions: Any) -> List[Dict[str, Any]]:
    """Parse tool descriptions, input: JSON text/list -> output: tool dict list"""
    if isinstance(tool_descriptions, str):
        try:
            tool_descriptions = json.loads(tool_descriptions)
        except json.JSONDecodeError:
            return []
    if not isinstance(tool_descriptions, list):
        return []
    return [tool for tool in tool_descriptions if isinstance(tool, dict)]


def _unique_tool_names(tool_names: Any) -> List[str]:
    """Dedupe tool names by order, input: names/list -> output: clean name list"""
    if isinstance(tool_names, str):
        tool_names = [tool_names]
    if not isinstance(tool_names, list):
        return []

    names = []
    seen = set()
    for name in tool_names:
        if not name:
            continue
        name = str(name)
        if name == "__reasoning__" or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def _get_tool_required_fields(tool: Dict[str, Any]) -> List[str]:
    """Get required fields, input: compact/full tool dict -> output: required field names"""
    required = tool.get("required")
    if not isinstance(required, list):
        parameters = tool.get("parameters")
        required = parameters.get("required") if isinstance(parameters, dict) else []
    return [str(field) for field in required] if isinstance(required, list) else []


def build_planner_tool_descriptions(tool_descriptions: Any) -> str:
    """Build planner tool view, input: full tools -> output: JSON name/description/required list"""
    tools = _load_tool_descriptions(tool_descriptions)
    seen = set()
    planner_tools = []
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        name = str(name)
        if name in seen:
            continue
        planner_tools.append({
            "name": name,
            "description": tool.get("description", ""),
            "required": _get_tool_required_fields(tool),
        })
        seen.add(name)
    return json.dumps(planner_tools, ensure_ascii=False, default=str)


def filter_tool_descriptions(tool_descriptions: Any, expected_tools: Any) -> str:
    """Filter executor tools, input: full tools/expected names -> output: JSON schemas"""
    tools = _load_tool_descriptions(tool_descriptions)
    tool_by_name = {}
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        name = str(name)
        if name not in tool_by_name:
            tool_by_name[name] = tool
    selected_tools = [
        tool_by_name[name]
        for name in _unique_tool_names(expected_tools)
        if name in tool_by_name
    ]
    return json.dumps(selected_tools, ensure_ascii=False, default=str)


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize tool calls, input: raw tool_calls -> output: list of tool call dicts"""
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        return []
    return [call for call in tool_calls if isinstance(call, dict)]


def _normalize_plan_steps(steps: Any) -> List[Dict[str, Any]]:
    """Normalize planner steps, input: raw steps -> output: step dict list"""
    if isinstance(steps, dict):
        steps = [steps]
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def _extract_visual_recognition_requests(parsed_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract planner visual requests, input: parsed plan -> output: normalized request list"""
    requests = []
    for step_idx, step in enumerate(_normalize_plan_steps(parsed_plan.get("steps", [])), 1):
        visual_request = step.get("visual_recognition_request")
        if not isinstance(visual_request, dict):
            continue

        request = {
            "step_id": step.get("step_id", step_idx),
            "visual_task": str(step.get("purpose", "")).strip(),
        }
        expected_key = str(visual_request.get("expected_key", "")).strip()
        if expected_key:
            request["expected_key"] = expected_key
        requests.append(request)
    return requests


def _normalize_visual_task(text: Any) -> str:
    """Normalize visual task text, input: any text -> output: comparable text"""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _visual_text_tokens(text: Any) -> set:
    """Tokenize visual text, input: any text -> output: comparable token set"""
    stopwords = {
        "a", "an", "analyze", "and", "as", "being", "box", "check", "confirm",
        "cookie", "cookies", "determine", "directly", "evaluate", "find", "for",
        "held", "i", "identify", "it", "item", "locate", "located", "me", "my",
        "name", "of", "on", "or", "packaged", "pick", "picked", "please",
        "product", "read", "the", "to", "up", "user", "verify", "visual", "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_visual_task(text))
        if token not in stopwords
    }


def _visual_text_matches(left: Any, right: Any) -> bool:
    """Match visual descriptions, input: two texts -> output: bool"""
    left_text = _normalize_visual_task(left)
    right_text = _normalize_visual_task(right)
    if not left_text or not right_text:
        return False
    if left_text in right_text or right_text in left_text:
        return True

    left_tokens = _visual_text_tokens(left_text)
    right_tokens = _visual_text_tokens(right_text)
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    return len(overlap) >= 2 and len(overlap) / min(len(left_tokens), len(right_tokens)) >= 0.5


def _visual_fact_matches_request(fact: Dict[str, Any], request: Dict[str, Any]) -> bool:
    """Check if a visual fact covers a request, input: fact/request -> output: bool"""
    if not isinstance(fact, dict) or not fact.get("name"):
        return False

    request_text = request.get("visual_task")
    return (
        _visual_text_matches(fact.get("target_description"), request_text) or
        _visual_text_matches(fact.get("task"), request_text)
    )


def _normalize_visual_facts(visual_facts: Any) -> List[Dict[str, Any]]:
    """Normalize visual facts, input: list/dict/none -> output: fact list"""
    if isinstance(visual_facts, list):
        return [fact for fact in visual_facts if isinstance(fact, dict)]
    if isinstance(visual_facts, dict):
        return [visual_facts]
    return []


def _pending_visual_recognition_requests(parsed_plan: Dict[str, Any], visual_facts: Optional[Any]) -> List[Dict[str, Any]]:
    """Find visual requests not already covered by facts, input: plan/facts -> output: pending requests"""
    current_facts = _normalize_visual_facts(visual_facts)
    pending = []
    for request in _extract_visual_recognition_requests(parsed_plan):
        if any(_visual_fact_matches_request(fact, request) for fact in current_facts):
            continue
        pending.append(request)
    return pending


def _build_reasoning_result(reasoning_result: Any, step_idx: int, step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Wrap executor reasoning, input: model result/step -> output: tool-result-shaped dict or None"""
    if reasoning_result is None or reasoning_result == "":
        return None
    if isinstance(reasoning_result, (dict, list)) and not reasoning_result:
        return None
    content = reasoning_result if isinstance(reasoning_result, (dict, list)) else {"result": reasoning_result}
    return {
        "role": "tool",
        "tool_name": "__reasoning__",
        "parameters": {
            "step_id": step.get("step_id", step_idx),
            "purpose": step.get("purpose", ""),
        },
        "content": json.dumps(content, ensure_ascii=False, default=str),
    }


def _decode_tool_result_content(content: Any) -> Any:
    """Decode tool content, input: JSON text/value -> output: parsed value or original text"""
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def build_executor_previous_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    """Build executor previous results, input: raw tool results -> output: JSON with decoded result"""
    decoded_results = []
    for result in tool_results:
        if not isinstance(result, dict):
            decoded_results.append({"result": result})
            continue
        decoded_results.append({
            "role": result.get("role"),
            "tool_name": result.get("tool_name"),
            "parameters": result.get("parameters", {}),
            "result": _decode_tool_result_content(result.get("content")),
        })
    return json.dumps(decoded_results, ensure_ascii=False, default=str)


def build_previous_tool_descriptions(tool_descriptions: Any, tool_results: List[Dict[str, Any]]) -> str:
    """Build schemas for previous tools, input: all tools/results -> output: JSON schemas"""
    previous_tool_names = [
        result.get("tool_name")
        for result in tool_results
        if isinstance(result, dict)
    ]
    return filter_tool_descriptions(tool_descriptions, previous_tool_names)


MAX_REPAIR_ROUNDS = 2
MAX_PLANNING_ROUNDS = 5
EXECUTOR_RESPONSE_MAX_ATTEMPTS = 3


def _extract_unresolved_items(reporter_response: str) -> List[str]:
    """Read reporter unresolved items, input: reporter JSON text -> output: unresolved list"""
    obj = _extract_json_object(reporter_response)
    if not obj:
        return []
    unresolved = obj.get("unresolved", [])
    if isinstance(unresolved, str):
        unresolved = [unresolved]
    if not isinstance(unresolved, list):
        return []
    return [str(item).strip() for item in unresolved if str(item).strip()]


def _build_repair_task(supervisor_task: str, unresolved_items: List[str]) -> str:
    """Build internal repair task, input: supervisor task/unresolved -> output: repair task"""
    items = "\n".join(f"- {item}" for item in unresolved_items)
    return (
        "Resolve only the unresolved issues below for the original Supervisor Task using real database tools. "
        "Do not ask the user, do not explain the repair, and do not repeat completed work "
        "when Previous Tool Results already provide the needed facts. "
        "If a database state change is missing, execute the real state-changing tool only "
        "when that state change is explicitly required by the original Supervisor Task. "
        f"Original Supervisor Task: {supervisor_task}\nUnresolved issues:\n{items}"
    )


def _bool_value(value: Any) -> bool:
    """Parse bool-like values, input: any value -> output: bool"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def run_tool_agent_plan_and_solve(
    db_instance,
    tool_descriptions: str,
    task: str,
    user_request: str,
    dialogue_history: List[Dict[str, Any]],
    visual_facts: Optional[Any],
) -> Dict[str, Any]:
    """Plan, execute, report, and repair unresolved tool work, input: task/context -> output: tool agent result"""
    full_tool_descriptions = _load_tool_descriptions(tool_descriptions)
    planner_tool_descriptions = build_planner_tool_descriptions(full_tool_descriptions)
    base_context = {
        "visual_facts": json.dumps(_normalize_visual_facts(visual_facts), ensure_ascii=False),
        "visual_expected_keys": json.dumps(get_visual_expected_keys(db_instance), ensure_ascii=False),
        "dialogue_history": _format_dialogue_history(dialogue_history),
    }

    all_tool_calls = []
    all_tool_results = []
    executor_responses = []
    planner_responses = []
    repair_logs = []
    plan_texts = []
    input_tokens = 0
    output_tokens = 0
    reporter_response = ""
    planner_call_round = 0

    def execute_plan_steps(parsed_plan: Dict[str, Any], planner_task: str, round_id: int) -> None:
        """Execute one planner plan, input: parsed plan/task/round id -> output: appends results"""
        nonlocal input_tokens, output_tokens
        steps = _normalize_plan_steps(parsed_plan.get("steps", []))

        for step_idx, step in enumerate(steps, 1):
            expected_tools = step.get("expected_tools") or []
            previous_tool_names = [
                result.get("tool_name")
                for result in all_tool_results
                if isinstance(result, dict)
            ]
            current_tool_names = _unique_tool_names(expected_tools)
            executor_tool_names = _unique_tool_names(current_tool_names + previous_tool_names)
            executor_prompt = TOOL_AGENT_EXECUTOR_PROMPT.format(
                **base_context,
                task=planner_task,
                plan=json.dumps(parsed_plan, ensure_ascii=False, default=str),
                current_step=json.dumps(step, ensure_ascii=False, default=str),
                previous_tool_results=build_executor_previous_tool_results(all_tool_results),
                tool_descriptions=filter_tool_descriptions(full_tool_descriptions, executor_tool_names),
            )
            executor_messages = [{"role": "user", "content": executor_prompt}]
            executor_response = ""
            parsed_executor = None
            executor_attempts = []
            parse_error = ""
            for attempt in range(1, EXECUTOR_RESPONSE_MAX_ATTEMPTS + 1):
                executor_profile = stage_latency.start("executor", model=TOOL_EXECUTOR_MODEL_NAME)
                try:
                    executor_response, step_input_tokens, step_output_tokens = call_tool_executor_model(
                        executor_messages
                    )
                except Exception:
                    stage_latency.end(executor_profile, status="error")
                    raise
                else:
                    stage_latency.end(executor_profile, step_input_tokens, step_output_tokens)
                input_tokens += step_input_tokens
                output_tokens += step_output_tokens

                response_text = str(executor_response or "")
                if not response_text.strip():
                    parse_error = "empty_response"
                    executor_attempts.append({
                        "attempt": attempt,
                        "response": executor_response,
                        "parse_error": parse_error,
                    })
                    continue

                parsed_executor = _extract_json_object(response_text)
                if parsed_executor is None:
                    parse_error = "invalid_json"
                    executor_attempts.append({
                        "attempt": attempt,
                        "response": executor_response,
                        "parse_error": parse_error,
                    })
                    continue

                parse_error = ""
                executor_attempts.append({
                    "attempt": attempt,
                    "response": executor_response,
                    "parse_error": "",
                })
                break

            executor_log = {
                "round": round_id,
                "step": step_idx,
                "response": executor_response,
            }
            if len(executor_attempts) > 1 or parse_error:
                executor_log["attempts"] = executor_attempts
            if parse_error:
                executor_log["parse_error"] = parse_error
            executor_responses.append({
                **executor_log,
            })

            if parsed_executor is None:
                continue
            step_tool_calls = _normalize_tool_calls(parsed_executor.get("tool_calls", []))
            reasoning_result = _build_reasoning_result(parsed_executor.get("reasoning_result"), step_idx, step)

            if step_tool_calls:
                step_results = execute_tool(db_instance, step_tool_calls)
                all_tool_calls.extend(step_tool_calls)
                all_tool_results.extend(step_results)
            if reasoning_result:
                all_tool_results.append(reasoning_result)

    def call_reporter() -> str:
        """Report current tool results, input: none -> output: reporter JSON text"""
        nonlocal input_tokens, output_tokens
        if not all_tool_results:
            return ""
        reporter_tool_names = [
            result.get("tool_name")
            for result in all_tool_results
            if isinstance(result, dict) and result.get("tool_name") != "__reasoning__"
        ]
        reporter_prompt = TOOL_AGENT_REPORTER_PROMPT.format(
            dialogue_history=base_context["dialogue_history"],
            visual_facts=base_context["visual_facts"],
            task=task,
            tool_results=build_executor_previous_tool_results(all_tool_results),
            tool_descriptions=filter_tool_descriptions(full_tool_descriptions, reporter_tool_names),
        )
        reporter_profile = stage_latency.start("reporter", model=TOOL_REPORTER_MODEL_NAME)
        try:
            response, reporter_input_tokens, reporter_output_tokens = call_tool_reporter_model(
                [{"role": "user", "content": reporter_prompt}]
            )
        except Exception:
            stage_latency.end(reporter_profile, status="error")
            raise
        else:
            stage_latency.end(reporter_profile, reporter_input_tokens, reporter_output_tokens)
        input_tokens += reporter_input_tokens
        output_tokens += reporter_output_tokens
        return response

    for planning_round in range(MAX_PLANNING_ROUNDS):
        planning_context = ""
        planner_descriptions = planner_tool_descriptions
        if planning_round > 0:
            planning_context = (
                "- Previous Tool Results: "
                f"{build_executor_previous_tool_results(all_tool_results)}\n"
                "- Previous Tool Descriptions: "
                f"{build_previous_tool_descriptions(full_tool_descriptions, all_tool_results)}\n"
            )

        planner_prompt = TOOL_AGENT_PLANNER_PROMPT.format(
            **base_context,
            task=task,
            repair_context=planning_context,
            repair_rules="",
            tool_descriptions=planner_descriptions,
        )
        planner_profile = stage_latency.start("planner", model=TOOL_PLANNER_MODEL_NAME)
        try:
            planner_response, planner_input_tokens, planner_output_tokens = call_tool_planner_model(
                [{"role": "user", "content": planner_prompt}]
            )
        except Exception:
            stage_latency.end(planner_profile, status="error")
            raise
        else:
            stage_latency.end(planner_profile, planner_input_tokens, planner_output_tokens)
        input_tokens += planner_input_tokens
        output_tokens += planner_output_tokens
        planner_responses.append({
            "round": planner_call_round,
            "round_type": "continuation" if planning_round > 0 else "planning",
            "task": task,
            "response": planner_response,
        })

        parsed_plan = _extract_json_object(planner_response) or {}
        plan_texts.append(parsed_plan.get("plan", str(planner_response)))
        pending_visual_requests = _pending_visual_recognition_requests(parsed_plan, visual_facts)
        if pending_visual_requests:
            raw_response_obj = {
                "planner": planner_responses[0]["response"] if planner_responses else "",
                "planners": planner_responses,
                "executors": executor_responses,
                "reporter": reporter_response,
                "repair": repair_logs,
                "visual_recognition_requests": pending_visual_requests,
            }
            raw_response = json.dumps(raw_response_obj, ensure_ascii=False, default=str)
            return {
                "plan": "\n".join(plan_texts),
                "tool_calls": all_tool_calls,
                "tool_results": all_tool_results,
                "raw_combined_result": "",
                "combined_result": "",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "raw_response": raw_response,
                "needs_visual_recognition": True,
                "visual_recognition_requests": pending_visual_requests,
            }
        execute_plan_steps(parsed_plan, task, planner_call_round)
        planner_call_round += 1

        if _bool_value(parsed_plan.get("requires_next_planning_round")) and planning_round < MAX_PLANNING_ROUNDS - 1:
            continue

        reporter_response = call_reporter()
        unresolved_items = _extract_unresolved_items(reporter_response)

        for repair_round in range(1, MAX_REPAIR_ROUNDS + 1):
            if not unresolved_items:
                break

            repair_task = _build_repair_task(task, unresolved_items)
            repair_logs.append({
                "round": planner_call_round,
                "unresolved": unresolved_items,
                "task": repair_task,
            })
            repair_context = (
                f"- Previous Tool Results: {build_executor_previous_tool_results(all_tool_results)}\n"
                "- Previous Tool Descriptions: "
                f"{build_previous_tool_descriptions(full_tool_descriptions, all_tool_results)}\n"
            )
            repair_rules = (
                "- This is a repair planning round. Use Previous Tool Results as completed context "
                "and plan only the missing real tool calls needed to resolve the unresolved issues. "
                "Do not set `requires_next_planning_round` to true during repair."
            )
            planner_prompt = TOOL_AGENT_PLANNER_PROMPT.format(
                **base_context,
                task=repair_task,
                repair_context=repair_context,
                repair_rules=repair_rules,
                tool_descriptions=planner_tool_descriptions,
            )
            planner_profile = stage_latency.start("planner", model=TOOL_PLANNER_MODEL_NAME)
            try:
                planner_response, planner_input_tokens, planner_output_tokens = call_tool_planner_model(
                    [{"role": "user", "content": planner_prompt}]
                )
            except Exception:
                stage_latency.end(planner_profile, status="error")
                raise
            else:
                stage_latency.end(planner_profile, planner_input_tokens, planner_output_tokens)
            input_tokens += planner_input_tokens
            output_tokens += planner_output_tokens
            planner_responses.append({
                "round": planner_call_round,
                "round_type": "repair",
                "task": repair_task,
                "response": planner_response,
            })

            parsed_plan = _extract_json_object(planner_response) or {}
            plan_texts.append(parsed_plan.get("plan", str(planner_response)))
            pending_visual_requests = _pending_visual_recognition_requests(parsed_plan, visual_facts)
            if pending_visual_requests:
                raw_response_obj = {
                    "planner": planner_responses[0]["response"] if planner_responses else "",
                    "planners": planner_responses,
                    "executors": executor_responses,
                    "reporter": reporter_response,
                    "repair": repair_logs,
                    "visual_recognition_requests": pending_visual_requests,
                }
                raw_response = json.dumps(raw_response_obj, ensure_ascii=False, default=str)
                return {
                    "plan": "\n".join(plan_texts),
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "raw_combined_result": "; ".join(res.get("content", str(res)) for res in all_tool_results),
                    "combined_result": "",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "raw_response": raw_response,
                    "needs_visual_recognition": True,
                    "visual_recognition_requests": pending_visual_requests,
                }
            execute_plan_steps(parsed_plan, repair_task, planner_call_round)
            planner_call_round += 1

            reporter_response = call_reporter()
            unresolved_items = _extract_unresolved_items(reporter_response)
        break

    result_strings = [res.get("content", str(res)) for res in all_tool_results]
    raw_combined_result = "; ".join(result_strings)
    reporter_summary = str(reporter_response).strip()

    unresolved_items = _extract_unresolved_items(reporter_response)
    if unresolved_items:
        reporter_obj = _extract_json_object(reporter_response)
        if reporter_obj:
            reporter_obj["unresolved"] = ["Some required tool actions could not be completed after retry."]
            reporter_response = json.dumps(reporter_obj, ensure_ascii=False, default=str)
            reporter_summary = reporter_response

    combined_result = raw_combined_result
    if reporter_summary and not reporter_summary.startswith("Error:"):
        combined_result = reporter_summary

    raw_response_obj = {
        "planner": planner_responses[0]["response"] if planner_responses else "",
        "planners": planner_responses,
        "executors": executor_responses,
        "reporter": reporter_response,
        "repair": repair_logs,
    }
    raw_response = json.dumps(raw_response_obj, ensure_ascii=False, default=str)
    return {
        "plan": "\n".join(plan_texts),
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
        "raw_combined_result": raw_combined_result,
        "combined_result": combined_result,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "raw_response": raw_response,
    }
