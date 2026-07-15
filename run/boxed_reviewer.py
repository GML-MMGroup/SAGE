"""Review and optionally correct one full boxed frame before crop generation."""
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from config.frames_review_config import (
    FRAMES_REVIEW_DEBUG_STDOUT,
    FRAMES_REVIEW_MODEL_NAME,
    FRAMES_REVIEW_SYSTEM_PROMPT,
    call_frames_review_model,
)
from run import stage_latency
from run.prompts import VISUAL_BOXED_REVIEW_PROMPT


def _extract_json_dict(text: Any) -> Dict[str, Any]:
    """Extract reviewer JSON, input: response text -> output: dict."""
    raw = str(text or "").strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _format_prompt(template: str, **values: Any) -> str:
    """Fill explicit reviewer prompt placeholders without touching JSON braces."""
    text = str(template)
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text.strip()


def _image_data_url(path: str) -> str:
    """Encode one boxed image, input: local path -> output: data URL."""
    mime_type, _ = mimetypes.guess_type(path)
    with open(path, "rb") as file:
        payload = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime_type or 'image/jpeg'};base64,{payload}"


def _normalize_review(data: Dict[str, Any], raw_response: Any) -> Dict[str, Any]:
    """Normalize fail-closed reviewer result, input: JSON/raw -> output: review dict."""
    verdict = str(data.get("verdict") or "").strip().lower()
    reason = str(data.get("reason") or "").strip()
    if verdict not in {"accept", "correct", "reject"}:
        return {
            "verdict": "reject",
            "corrected_bbox": None,
            "reason": reason or "invalid_or_missing_reviewer_verdict",
            "model": FRAMES_REVIEW_MODEL_NAME,
            "raw_response": str(raw_response or "").strip(),
        }
    corrected_bbox = data.get("corrected_bbox")
    if verdict != "correct":
        corrected_bbox = None
    elif not isinstance(corrected_bbox, (list, tuple)) or len(corrected_bbox) < 4:
        verdict = "reject"
        corrected_bbox = None
        reason = reason or "missing_corrected_bbox"
    return {
        "verdict": verdict,
        "corrected_bbox": list(corrected_bbox[:4]) if corrected_bbox is not None else None,
        "reason": reason,
        "model": FRAMES_REVIEW_MODEL_NAME,
        "raw_response": str(raw_response or "").strip(),
    }


def review_boxed_frame(boxed_frame: Dict[str, Any], target_text: str) -> Dict[str, Any]:
    """Review one boxed full frame, input: boxed metadata/task -> output: normalized verdict."""
    image_path = str(boxed_frame.get("path") or "")
    boxes = boxed_frame.get("boxes") or []
    if not image_path or not Path(image_path).exists() or not boxes:
        return _normalize_review({}, "missing_boxed_frame_or_bbox")
    original_bbox = boxes[0].get("box") if isinstance(boxes[0], dict) else None
    if not original_bbox:
        return _normalize_review({}, "missing_original_bbox")
    with Image.open(image_path) as image:
        width, height = image.size
    prompt = _format_prompt(
        VISUAL_BOXED_REVIEW_PROMPT,
        target_text=str(target_text or "").strip(),
        width=width,
        height=height,
        second=boxed_frame.get("second"),
        original_bbox=json.dumps(original_bbox, ensure_ascii=False),
        candidate_desc=str(boxes[0].get("desc") or ""),
    )
    messages = [
        {"role": "system", "content": FRAMES_REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
        ]},
    ]
    reviewer_profile = stage_latency.start("frames_reviewer", model=FRAMES_REVIEW_MODEL_NAME)
    try:
        response, input_tokens, output_tokens = call_frames_review_model(messages)
    except Exception:
        stage_latency.end(reviewer_profile, status="error")
        raise
    else:
        response_status = "error" if str(response or "").strip().startswith("Error:") else "success"
        stage_latency.end(
            reviewer_profile,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status=response_status,
        )
    if FRAMES_REVIEW_DEBUG_STDOUT:
        print(f"[frames_review debug raw response] {str(response).strip()}", flush=True)
    return _normalize_review(_extract_json_dict(response), response)
