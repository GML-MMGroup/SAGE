"""
Supervisor helpers for service-side multi-agent delegation.
"""
import json
import re
from typing import Any, Dict, Optional

from config.service_agent_config import SUPERVISOR_AGENT_SYSTEM_PROMPT
from run.prompts import SUPERVISOR_AGENT_PROMPT_BASE


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

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


def build_supervisor_prompt(tool_descriptions: str = "") -> str:
    """Build supervisor system prompt, input: unused tool descriptions -> output: prompt"""
    return f"{SUPERVISOR_AGENT_SYSTEM_PROMPT}\n\n{SUPERVISOR_AGENT_PROMPT_BASE.strip()}"


def parse_supervisor_action(response_text: str, fallback_task: str = "") -> Dict[str, Any]:
    """Parse supervisor action, input: response text/fallback task -> output: two-field action dict"""
    obj = _extract_json_object(response_text)
    if obj:
        agent_name = obj.get("agent_name")
        task = obj.get("task")
        if agent_name in {"visual_agent", "tool_agent", "ask_user"} and isinstance(task, str) and task.strip():
            return {"agent_name": agent_name, "task": task.strip()}
    return {"agent_name": "tool_agent", "task": fallback_task or str(response_text).strip()}


def _clean_items(value: Any) -> list:
    """Normalize response fields, input: scalar/list -> output: non-empty text list"""
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _as_sentence(text: str) -> str:
    """Ensure sentence punctuation, input: text -> output: sentence"""
    text = str(text).strip()
    if not text:
        return ""
    return text if text[-1] in ".!?" else f"{text}."


def format_agent_result_for_user(response_text: str) -> str:
    """Render internal agent JSON as user text, input: agent result -> output: natural reply"""
    text = str(response_text).strip()
    obj = _extract_json_object(text)
    if not obj:
        return text

    task_answer = str(obj.get("task_answer", "")).strip()
    if task_answer:
        return task_answer

    if not any(key in obj for key in ("facts", "state_changes", "calculations", "unresolved")):
        return text

    confirmed = []
    for key in ("facts", "state_changes", "calculations"):
        confirmed.extend(_clean_items(obj.get(key)))
    unresolved = _clean_items(obj.get("unresolved"))

    parts = [_as_sentence(item) for item in confirmed if item]
    if unresolved:
        if parts:
            parts.append("I'm sorry, but I couldn't confirm " + "; ".join(unresolved) + ".")
        else:
            parts.append("I'm sorry, but I couldn't confirm that from the available information. " + " ".join(_as_sentence(item) for item in unresolved))
    return " ".join(part for part in parts if part).strip() or text
