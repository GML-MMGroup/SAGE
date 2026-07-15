"""
Visual recognition agent for service-side item identification.
"""
import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from config.visual_agent_config import VISUAL_AGENT_SYSTEM_PROMPT, VISUAL_IMAGE_DETAIL, call_visual_agent_model
from config.visual_agent_config import VISUAL_AGENT_MODEL_NAME
from config.visual_boxed_config import VISUAL_BOXED_SCENARIO_PREFIXES
from run.prompts import VISUAL_RECOGNITION_AGENT_PROMPT
from run import stage_latency
from run.utils import build_message_with_media


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from text, input: model text -> output: dict or None"""
    text = str(text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
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


def _normalize_text(value: Any) -> str:
    """Normalize candidate text, input: any value -> output: lowercase stripped string"""
    return str(value or "").strip().lower()


def _build_target_description(task_text: Any) -> str:
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


def _format_visual_task_for_prompt(task_text: Any, db_instance) -> str:
    """Build recognizer task wording, input: task/db -> output: prompt task text"""
    task = str(task_text or "").strip()
    if getattr(db_instance, "vision_video_model", "image_base64") != "image_base64":
        return task
    clean_task = _build_target_description(task)
    scenario = str(getattr(db_instance, "vision_scenario", "") or "").lower()
    scenario_box_enabled = any(scenario.startswith(prefix) for prefix in VISUAL_BOXED_SCENARIO_PREFIXES)
    box_enabled = bool(getattr(db_instance, "vision_box_enabled", True)) and scenario_box_enabled
    if not box_enabled:
        return (
            "Please identify the visual target using the selected frames. "
            f"Original task: {clean_task}. "
            "Selected frames may separately show the physical target and its textual representation. Combine them only when same-frame association, "
            "position/layout, temporal continuity, or explicit pairing confirms the same entity; do not borrow text from a nearby item, another row/page, "
            "or another menu. Use visible actions, positions, labels, colors, shapes, and appearance cues in the provided images."
        )
    return (
        "Please identify the visual target using the processed frames and any green Target boxes. "
        f"Original task: {clean_task}. "
        "Selected frames may separately show the physical target and its textual representation. Combine them only when same-frame association, "
        "position/layout, temporal continuity, or explicit pairing confirms the same entity; do not borrow text from a nearby item, another row/page, "
        "or another menu. "
        "A green Target box marks the target requested by the current task, which may be a visible physical item or a target presented as text or a label. Still verify the green Target box against the task wording and visible context; it is not final ground truth. "
        "When the target inside the green Target box clearly corresponds to the original task and visible context, prioritize identifying the target inside the green Target box. "
        "If the boxed region clearly conflicts with stronger visible evidence or the task constraints, identify the target that best satisfies the full visual task. "
        "If no green Target box is present, use the original task and provided images, including any visible color or appearance cues, to identify the target."
    )


def _add_candidate(candidates: List[Dict[str, Any]], seen: set, key: str, name: Any, restaurant_name: Any = None, details: Optional[Dict[str, Any]] = None) -> None:
    """Add one visual candidate, input: candidate fields -> output: none"""
    item_name = _normalize_text(name)
    if not item_name:
        return
    restaurant = str(restaurant_name).strip() if restaurant_name else None
    candidate_id = f"{key}::{_normalize_text(restaurant)}::{item_name}" if restaurant else f"{key}::{item_name}"
    if candidate_id in seen:
        return
    seen.add(candidate_id)
    candidate = {
        "candidate_id": candidate_id,
        "key": key,
        "name": item_name,
        "restaurant_name": restaurant,
    }
    if details:
        candidate["details"] = details
    candidates.append(candidate)


def _build_visual_candidates(db_instance) -> List[Dict[str, Any]]:
    """Build visual candidates from the current scenario DB, input: db -> output: candidate dicts"""
    candidates: List[Dict[str, Any]] = []
    seen = set()

    if getattr(db_instance, "restaurants", None):
        for restaurant_name, store in db_instance.restaurants.items():
            for dish in store.get("catalog", {}).values():
                _add_candidate(candidates, seen, "dish_name", dish.name, restaurant_name, {"category": getattr(dish, "category", "")})
                _add_candidate(candidates, seen, "category", getattr(dish, "category", ""), restaurant_name)
            for set_meal in store.get("set_meals", {}).values():
                included = [item.get("dish_name") for item in getattr(set_meal, "included_dishes", [])]
                _add_candidate(candidates, seen, "set_meal_name", set_meal.name, restaurant_name, {"included_dishes": included})
        return candidates

    if getattr(db_instance, "ingredients", None) or getattr(db_instance, "recipes", None):
        for ingredient in getattr(db_instance, "ingredients", {}).values():
            _add_candidate(
                candidates, seen, "ingredient_name", ingredient.name,
                details={"category": getattr(ingredient, "category", ""), "storage_location": getattr(ingredient, "storage_location", "")}
            )
        for recipe in getattr(db_instance, "recipes", {}).values():
            ingredients = [getattr(item, "ingredient_name", "") for item in getattr(recipe, "ingredients", [])]
            _add_candidate(candidates, seen, "recipe_name", recipe.name, details={"ingredients": ingredients})
        return candidates

    if getattr(db_instance, "catalog", None):
        for item in db_instance.catalog.values():
            if hasattr(item, "country_of_origin"):
                _add_candidate(candidates, seen, "product_name", item.name, details={"category": getattr(item, "category", "")})
            else:
                _add_candidate(candidates, seen, "dish_name", item.name, details={"category": getattr(item, "category", "")})
                _add_candidate(candidates, seen, "category", getattr(item, "category", ""))
        for set_meal in getattr(db_instance, "set_meals", {}).values():
            included = [item.get("dish_name") for item in getattr(set_meal, "included_dishes", [])]
            _add_candidate(candidates, seen, "set_meal_name", set_meal.name, details={"included_dishes": included})

    return candidates


def _format_candidate_text(candidates: List[Dict[str, Any]]) -> str:
    """Format candidate list for the visual prompt, input: candidates -> output: JSONL string"""
    return "\n".join(json.dumps(candidate, ensure_ascii=False) for candidate in candidates)


def build_visual_catalog(db_instance) -> List[Dict[str, Any]]:
    """Build visual catalog, input: db -> output: candidate list"""
    return _build_visual_candidates(db_instance)


def get_visual_expected_keys(db_instance) -> List[str]:
    """Get visual candidate keys, input: db -> output: ordered key list"""
    keys = []
    for candidate in _build_visual_candidates(db_instance):
        key = candidate.get("key")
        if key and key not in keys:
            keys.append(key)
    return keys


def _match_value_to_candidate(value: Any, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match scenario value to catalog candidate, input: value/candidates -> output: candidate or None"""
    value_text = _normalize_text(value)
    if not value_text:
        return None
    direct_matches = [c for c in candidates if value_text == _normalize_text(c.get("name"))]
    if direct_matches:
        return direct_matches[0]
    contains_matches = [c for c in candidates if value_text in _normalize_text(c.get("name")) or _normalize_text(c.get("name")) in value_text]
    if contains_matches:
        return sorted(contains_matches, key=lambda c: len(_normalize_text(c.get("name"))), reverse=True)[0]
    names = [_normalize_text(c.get("name")) for c in candidates]
    close = difflib.get_close_matches(value_text, names, n=1, cutoff=0.55)
    if close:
        return candidates[names.index(close[0])]
    return None


def resolve_expected_visual_names(db_instance, scenario: Dict[str, Any]) -> List[str]:
    """Resolve expected visual names, input: db/scenario -> output: full catalog names"""
    candidates = _build_visual_candidates(db_instance)
    key = scenario.get("key")
    values = scenario.get("value", [])
    if isinstance(values, str):
        values = [values]
    key_candidates = [c for c in candidates if c.get("key") == key] or candidates
    resolved = []
    for value in values:
        match = _match_value_to_candidate(value, key_candidates)
        resolved.append(match["name"] if match else _normalize_text(value))
    return resolved


def _match_candidate_fields(parsed: Dict[str, Any], candidates: List[Dict[str, Any]], candidate_by_id: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match parsed candidate fields, input: parsed/candidates/id map -> output: candidate or None"""
    candidate_id = _normalize_text(parsed.get("candidate_id"))
    if candidate_id in candidate_by_id:
        return candidate_by_id[candidate_id]

    key = _normalize_text(parsed.get("key"))
    name = _normalize_text(parsed.get("name"))
    for fallback_key in ["product_name", "dish_name", "recipe_name", "ingredient_name", "category", "set_meal_name"]:
        if parsed.get(fallback_key):
            key = key or fallback_key
            name = name or _normalize_text(parsed.get(fallback_key))
            break

    restaurant_name = _normalize_text(parsed.get("restaurant_name"))
    matches = []
    for candidate in candidates:
        if key and candidate.get("key") != key:
            continue
        if name and _normalize_text(candidate.get("name")) != name:
            continue
        if restaurant_name and _normalize_text(candidate.get("restaurant_name")) != restaurant_name:
            continue
        matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    return None


def _match_candidate(response_text: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match model output to one candidate, input: model text/candidates -> output: candidate or None"""
    candidate_by_id = {candidate["candidate_id"].lower(): candidate for candidate in candidates}
    response_value = _normalize_text(response_text.strip("`'\"* \n\t"))
    if response_value in candidate_by_id:
        return candidate_by_id[response_value]

    same_name = [candidate for candidate in candidates if _normalize_text(candidate.get("name")) == response_value]
    if len(same_name) == 1:
        return same_name[0]

    response_norm = _normalize_text(response_text)
    name_in_response = []
    for candidate in candidates:
        name = _normalize_text(candidate.get("name"))
        if not name:
            continue
        if len(name) == 1:
            if re.search(rf"(?<![a-z0-9'’]){re.escape(name)}(?![a-z0-9'’])", response_norm):
                name_in_response.append(candidate)
        elif name in response_norm:
            name_in_response.append(candidate)
    if name_in_response:
        return sorted(name_in_response, key=lambda c: len(_normalize_text(c.get("name"))), reverse=True)[0]

    parsed = _extract_json_object(response_text)
    if parsed:
        return _match_candidate_fields(parsed, candidates, candidate_by_id)

    generic = {
        "wine", "bottle", "item", "product", "the", "referenced",
        "pinot", "noir", "merlot", "sauvignon", "blanc", "chardonnay", "reserve",
    }
    response_tokens = set(re.findall(r"[a-z0-9]+", response_norm)) - generic
    scored = []
    for candidate in candidates:
        name_tokens = set(re.findall(r"[a-z0-9]+", _normalize_text(candidate.get("name")))) - generic
        overlap = response_tokens & name_tokens
        if len(overlap) >= 2:
            scored.append((len(overlap), len(name_tokens), candidate))
    if scored:
        return sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)[0][2]
    return None


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate candidates, input: candidate list -> output: candidate list"""
    deduped = []
    seen = set()
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        deduped.append(candidate)
    return deduped


def _match_candidates(response_text: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Match model output to candidates, input: model text/candidates -> output: candidate list"""
    candidate_by_id = {candidate["candidate_id"].lower(): candidate for candidate in candidates}
    parsed = _extract_json_object(response_text)
    matched = []

    if parsed:
        raw_matches = parsed.get("matches")
        if isinstance(raw_matches, dict):
            raw_matches = [raw_matches]
        if isinstance(raw_matches, list):
            for item in raw_matches:
                if not isinstance(item, dict):
                    continue
                candidate = _match_candidate_fields(item, candidates, candidate_by_id)
                if candidate:
                    matched.append(candidate)
            matched = _dedupe_candidates(matched)
            if matched:
                return matched

        candidate = _match_candidate_fields(parsed, candidates, candidate_by_id)
        if candidate:
            return [candidate]

    candidate = _match_candidate(response_text, candidates)
    return [candidate] if candidate else []


def _candidate_to_result(candidate: Dict[str, Any], task: str, attempt_idx: int, response_text: str = "") -> Dict[str, Any]:
    """Convert a candidate to visual result, input: candidate/task/attempt/response -> output: result dict"""
    key = candidate["key"]
    natural_result = f'The referenced item is "{candidate["name"]}".'
    result = {
        "status": "success",
        "task": task,
        "target_description": _build_target_description(task),
        "result": natural_result,
        "raw_response": str(response_text or "").strip(),
        "candidate_id": candidate["candidate_id"],
        "key": key,
        "name": candidate["name"],
        "attempt": attempt_idx,
        key: candidate["name"],
    }
    if candidate.get("restaurant_name"):
        result["restaurant_name"] = candidate["restaurant_name"]
    return result


def _candidates_to_result(candidates: List[Dict[str, Any]], task: str, attempt_idx: int, response_text: str = "") -> Dict[str, Any]:
    """Convert candidates to visual result, input: candidates/task/attempt/response -> output: result dict"""
    if len(candidates) == 1:
        return _candidate_to_result(candidates[0], task, attempt_idx, response_text)

    names = [candidate["name"] for candidate in candidates]
    keys = [candidate["key"] for candidate in candidates]
    natural_result = "The referenced items are " + ", ".join(f'"{name}"' for name in names) + "."
    result = {
        "status": "success",
        "task": task,
        "target_description": _build_target_description(task),
        "result": natural_result,
        "raw_response": str(response_text or "").strip(),
        "candidate_id": [candidate["candidate_id"] for candidate in candidates],
        "key": keys,
        "name": names,
        "attempt": attempt_idx,
    }
    for key in sorted(set(keys)):
        key_names = [candidate["name"] for candidate in candidates if candidate.get("key") == key]
        result[key] = key_names
    restaurant_names = [candidate.get("restaurant_name") for candidate in candidates]
    if any(restaurant_names):
        result["restaurant_name"] = restaurant_names
    return result


def _build_visual_content(text: str, media_path: str, db_instance, target: Optional[str] = None, visual_facts: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Build visual input, input: text/media/db -> output: multimodal content list"""
    return build_message_with_media(
        text,
        media_path,
        use_vision=True,
        service_model_name=getattr(db_instance, "vision_service_model_name", "qwen3-vl-225b"),
        image_detail=VISUAL_IMAGE_DETAIL,
        scenario=getattr(db_instance, "vision_scenario", None),
        target=target,
        task_id=getattr(db_instance, "vision_task_id", None),
        request_seq=getattr(db_instance, "vision_request_seq", None),
        box_enabled=getattr(db_instance, "vision_box_enabled", True),
        visual_facts=visual_facts,
    )


def run_visual_recognition_agent(db_instance, task: str, expected_key: Optional[str] = None, visual_facts: Optional[Any] = None) -> Tuple[Dict[str, Any], int, int]:
    """Identify item from visual context, input: db/task/optional key -> output: result and token counts"""
    if not getattr(db_instance, "vision_media_path", None):
        return {"status": "error", "task": task, "message": "No visual context is available for visual recognition."}, 0, 0

    candidates = _build_visual_candidates(db_instance)
    if not candidates:
        return {"status": "error", "task": task, "message": "No visual candidates are available for this scenario."}, 0, 0

    expected_key = _normalize_text(expected_key)
    if expected_key:
        candidates = [candidate for candidate in candidates if candidate.get("key") == expected_key]
        if not candidates:
            return {
                "status": "error",
                "task": task,
                "message": f"No visual candidates are available for expected key: {expected_key}."
            }, 0, 0
    db_instance.vision_request_seq = int(getattr(db_instance, "vision_request_seq", 0) or 0) + 1

    candidate_text = _format_candidate_text(candidates)
    expected_key_text = ""
    if expected_key:
        expected_key_text = (
            f"Expected catalog key: {expected_key}\n"
            "Use this expected key as a hard constraint. If the natural-language task suggests another "
            "item type, still choose only from the candidates under the expected catalog key.\n\n"
        )
    visual_task = _format_visual_task_for_prompt(task, db_instance)
    prompt_tail = (
        f"{expected_key_text}"
        "Candidate catalog:\n"
        f"{candidate_text}\n\n"
        f"{VISUAL_RECOGNITION_AGENT_PROMPT.strip()}"
    )
    if getattr(db_instance, "vision_video_model", "image_base64") == "image_base64":
        user_content = _build_visual_content(
            visual_task,
            db_instance.vision_media_path,
            db_instance,
            target=task,
            visual_facts=visual_facts,
        )
        user_content.append({"type": "text", "text": prompt_tail})
    else:
        prompt = (
            f"{visual_task}\n\n"
            f"{prompt_tail}"
        )
        user_content = _build_visual_content(
            prompt,
            db_instance.vision_media_path,
            db_instance,
            target=task,
            visual_facts=visual_facts,
        )
    messages = [
        {"role": "system", "content": VISUAL_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    visual_profile = stage_latency.start("visual_recognition", model=VISUAL_AGENT_MODEL_NAME)
    try:
        response, input_tokens, output_tokens = call_visual_agent_model(messages)
    except Exception:
        stage_latency.end(visual_profile, status="error")
        raise
    else:
        stage_latency.end(visual_profile, input_tokens, output_tokens)

    response_text = str(response).strip()
    if _normalize_text(response_text.strip("`'\"* \n\t")) != "unknown":
        matched_candidates = _match_candidates(response_text, candidates)
        if matched_candidates:
            return _candidates_to_result(matched_candidates, task, 1, response_text), input_tokens, output_tokens

    return {
        "status": "error",
        "task": task,
        "result": response_text,
        "stop_required": True,
        "attempts": 1,
        "message": "Visual recognition failed. Ask the user to reply exactly STOP."
    }, input_tokens, output_tokens
