"""
Generic boxed-frame layer for image_base64 visual inputs.

Execution:
1. Keep the frame selecter's selected frames as source evidence.
2. Resolve target product boxes with a generic VLM locator.
3. Keep only confident locator candidates and select final target boxes.
4. Copy auxiliary selected frames unchanged after boxed evidence.
"""
import json
import base64
import math
import mimetypes
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from config.visual_boxed_config import (
    VISUAL_BOXED_DEBUG_STDOUT,
    VISUAL_BOXED_MAX_RETRIES,
    VISUAL_BOXED_MODEL_NAMES,
    VISUAL_BOXED_SCENARIO_PREFIXES,
    VISUAL_BOXED_SYSTEM_PROMPT,
    call_visual_boxed_model,
)
from run.boxed_reviewer import review_boxed_frame
from run.prompts import VISUAL_BOXED_LOCATOR_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOX_ROOT = PROJECT_ROOT / "processed" / "frame_select_boxed"

BOX_VERSION = "visual_box_v13_system_grounding_rules"


def _scenario_enabled(scenario: Any) -> bool:
    """Check boxed grounding scope, input: scenario -> output: bool."""
    label = str(scenario or "").strip().lower()
    return any(label.startswith(prefix) for prefix in VISUAL_BOXED_SCENARIO_PREFIXES)


def _clean_target(text: Any) -> str:
    """Strip visual wrappers, input: text -> output: target text."""
    s = str(text or "").strip()
    s = re.sub(r"(?is)^\s*please\s+identify\s+the\s+visual\s+target\s*\.?\s*task\s*:\s*", "", s).strip()
    return s.rstrip(".")


def _normalize_visual_facts_for_prompt(visual_facts: Any) -> List[Dict[str, Any]]:
    """Normalize visual facts for prompt context, input: facts -> output: list."""
    if isinstance(visual_facts, list):
        return [fact for fact in visual_facts if isinstance(fact, dict)]
    if isinstance(visual_facts, dict):
        return [visual_facts] if visual_facts else []
    return []


def _format_previous_visual_facts_context(visual_facts: Any) -> str:
    """Format previous visual facts, input: facts -> output: prompt text."""
    facts = _normalize_visual_facts_for_prompt(visual_facts)
    if not facts:
        return ""
    return (
        "Previous visual facts:\n"
        "Use these only if the current visual task refers to a previous, initial, selected, same, "
        "adjacent, left/right, above/below, or already identified target; otherwise ignore them.\n"
        "These are previously recognized visual targets, not new targets to identify.\n\n"
        f"{json.dumps(facts, ensure_ascii=False, default=str)}"
    )


def _source_signature(
    selected_infos: List[Dict[str, Any]],
    auxiliary_infos: List[Dict[str, Any]],
    target_text: str,
    scenario: Any,
    target_cardinality: Dict[str, Any],
) -> Dict[str, Any]:
    """Build cache signature, input: selected frames/target -> output: dict."""
    sources = []
    for item in selected_infos:
        path = str(item.get("path") or "")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        sources.append({"path": path, "second": item.get("second"), "mtime": round(mtime, 3)})
    auxiliary_sources = []
    for item in auxiliary_infos:
        path = str(item.get("path") or "")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        auxiliary_sources.append({"path": path, "second": item.get("second"), "mtime": round(mtime, 3)})
    return {
        "version": BOX_VERSION,
        "scenario": str(scenario or ""),
        "target": str(target_text or ""),
        "target_cardinality": target_cardinality,
        "sources": sources,
        "auxiliary_sources": auxiliary_sources,
    }


def _cached_manifest_has_nonconfident_box(manifest: Dict[str, Any]) -> bool:
    """Reject cached rendered boxes without confident locator evidence."""
    for key in ("boxed_main_frames", "sent_frames", "boxed_frames"):
        frames = manifest.get(key) or []
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for box in frame.get("boxes") or []:
                if isinstance(box, dict) and box.get("locator_certainty") != "confident":
                    return True
    return False


def _load_cached(save_dir: Path, signature: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Load cached boxed frames, input: save dir/signature -> output: infos/meta or none."""
    manifest_path = save_dir / "manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if manifest.get("signature") != signature:
        return None, None
    if _cached_manifest_has_nonconfident_box(manifest):
        return None, None
    infos = []
    for item in manifest.get("sent_frames") or manifest.get("boxed_frames") or []:
        path = item.get("path")
        if path and os.path.exists(path):
            infos.append({
                "path": path,
                "second": item.get("second"),
                "kind": item.get("kind"),
                "paired_anchor_second": item.get("paired_anchor_second"),
                "source_path": item.get("source_path"),
            })
    if not infos:
        return None, None
    return infos, manifest


def _clear_outputs(save_dir: Path) -> None:
    """Clear request outputs before regeneration, input: save dir -> output: none."""
    if not save_dir.exists():
        return
    for path in save_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _request_source_dir(final_scenario: Any, final_task_id: Any, request_seq: int) -> Path:
    """Build selected-frame cache dir, input: scenario/task/request -> output: path."""
    scenario = str(final_scenario or "unknown")
    task_label = f"task{int(final_task_id)}" if final_task_id is not None else "task_unknown"
    return PROJECT_ROOT / "processed" / "frame_select" / scenario / task_label / f"request_{request_seq:03d}"


def _repair_selected_paths(selected_infos: List[Dict[str, Any]], source_dir: Path) -> List[Dict[str, Any]]:
    """Map stale selected paths to local cache files, input: infos/source dir -> output: infos."""
    repaired = []
    for item in selected_infos:
        info = dict(item)
        path = Path(str(info.get("path") or ""))
        if path.exists():
            repaired.append(info)
            continue
        candidates = []
        if path.name:
            candidates.append(source_dir / path.name)
        second = info.get("second")
        if second is not None:
            try:
                candidates.extend(source_dir.glob(f"selected_*_frame_{int(second):04d}s.*"))
            except Exception:
                pass
        for candidate in candidates:
            if candidate.exists():
                info["path"] = str(candidate)
                break
        repaired.append(info)
    return repaired


def _image_size(path: str) -> Tuple[int, int]:
    """Read image size, input: path -> output: width/height."""
    with Image.open(path) as image:
        return image.size


def _b64(path: str) -> str:
    """Encode image as base64, input: path -> output: base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime_type(path: str) -> str:
    """Get image MIME type, input: path -> output: MIME type."""
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "image/jpeg"


def _format_prompt(template: str, **values: Any) -> str:
    """Replace explicit prompt placeholders without touching JSON braces."""
    text = str(template)
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text.strip()


def _normalize_box(box: Any, width: int, height: int) -> Optional[List[float]]:
    """Clamp bbox to image bounds, input: box/size -> output: box or None."""
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
    except Exception:
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    area = (x2 - x1) * (y2 - y1) / max(width * height, 1)
    if area < 0.003 or area > 0.92:
        return None
    return [x1, y1, x2, y2]


def _extract_json_dict(text: Any) -> Dict[str, Any]:
    """Extract first JSON object, input: model text -> output: dict."""
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


def _model_call(content: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Call grounding VLMs with retries, input: content -> output: JSON/model."""
    messages = [
        {"role": "system", "content": VISUAL_BOXED_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    for model in VISUAL_BOXED_MODEL_NAMES:
        try:
            response, _, _ = call_visual_boxed_model(
                messages,
                model_name=model,
                max_retries=VISUAL_BOXED_MAX_RETRIES,
                enable_thinking=None,
            )
            if VISUAL_BOXED_DEBUG_STDOUT:
                print(f"[visual_box debug raw response] {model}: {str(response).strip()}", flush=True)
            data = _extract_json_dict(response)
            if data:
                return data, model
        except Exception as exc:
            print(f"[visual_box] VLM retry failed: {model} {str(exc)[:120]}", flush=True)
    return {}, None


def _locate_selected(
    selected_infos: List[Dict[str, Any]],
    target_text: str,
    cardinality: Dict[str, Any],
    visual_facts: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Locate candidates across selected frames, input: selected infos/target -> output: candidates."""
    frames = []
    for idx, info in enumerate(selected_infos, 1):
        path = str(info.get("path") or "")
        if not path or not os.path.exists(path):
            continue
        width, height = _image_size(path)
        frames.append({
            "idx": idx,
            "path": path,
            "second": info.get("second"),
            "width": width,
            "height": height,
        })
    if not frames:
        return []

    frame_context = "\n".join(
        f"- Frame {item['idx']}: second {item.get('second')}, resolution {item['width']}x{item['height']}, "
        f"use frame_index={item['idx']} for bboxes on this image."
        for item in frames
    )
    prompt = _format_prompt(
        VISUAL_BOXED_LOCATOR_PROMPT,
        frame_context=frame_context,
        target_text=_clean_target(target_text),
        target_cardinality=cardinality.get("cardinality", "single"),
        max_targets=cardinality.get("max_targets", 1),
    )
    previous_visual_facts_context = _format_previous_visual_facts_context(visual_facts)
    if previous_visual_facts_context:
        prompt = f"{prompt}\n\n{previous_visual_facts_context}"
    content = [{"type": "text", "text": prompt}]
    for item in frames:
        content.append({
            "type": "text",
            "text": (
                f"Frame {item['idx']} image. second {item.get('second')}, "
                f"resolution {item['width']}x{item['height']}."
            ),
        })
        content.append({"type": "image_url", "image_url": {"url": f"data:{_mime_type(item['path'])};base64,{_b64(item['path'])}"}})

    data, model = _model_call(content)
    targets = data.get("targets")
    if isinstance(targets, dict):
        targets = [targets]
    if not isinstance(targets, list):
        targets = []

    frame_by_idx = {item["idx"]: item for item in frames}
    candidates = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        try:
            frame_index = int(target.get("frame_index"))
        except Exception:
            continue
        frame = frame_by_idx.get(frame_index)
        if not frame:
            continue
        box = _normalize_box(target.get("bbox"), int(frame["width"]), int(frame["height"]))
        if not box:
            continue
        certainty = str(target.get("certainty") or "").strip().lower()
        if certainty not in {"confident", "uncertain"}:
            certainty = "uncertain"
        candidates.append({
            "idx": frame_index,
            "path": frame["path"],
            "second": frame.get("second"),
            "size": (frame["width"], frame["height"]),
            "box": box,
            "desc": str(target.get("desc") or ""),
            "evidence": str(target.get("evidence") or "other"),
            "locator_certainty": certainty,
            "model": model,
        })

    return candidates


def _target_has_action_constraint(target_text: str) -> bool:
    """Check whether task requires an action cue, input: target text -> output: bool."""
    target = _clean_target(target_text).lower()
    return bool(re.search(
        r"\b(point(?:ed|ing)?|tap(?:ped|ping)?|pick(?:ed|ing)?|picked\s+up|held|holding|put\s+down|"
        r"placed|selected|selecting|touch(?:ed|ing)?|circl(?:ed|ing))\b",
        target,
    ))


def _candidate_action_score(candidate: Dict[str, Any]) -> int:
    """Score action evidence in a candidate, input: candidate -> output: 0/1."""
    text = str(candidate.get("evidence") or "").lower()
    return int(bool(re.search(
        r"\b(point\w*|tap\w*|pick\w*|held|holding|put\s+down|placed|select\w*|touch\w*|circl\w*|action|gesture)\b",
        text,
    )))


def _select_final_candidates(
    candidates: List[Dict[str, Any]],
    target_text: str,
    cardinality: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Select final target candidates, input: candidates/target -> output: candidates."""
    if not candidates:
        return []
    action_required = _target_has_action_constraint(target_text)
    ordered = sorted(
        candidates,
        key=lambda item: _candidate_action_score(item) if action_required else 0,
        reverse=True,
    )
    if cardinality.get("cardinality") == "multiple":
        try:
            limit = int(cardinality.get("max_targets") or 6)
        except Exception:
            limit = 6
        return ordered[:max(1, min(limit, 6))]
    return ordered[:1]


def _draw_boxes(source_path: str, boxes: List[Dict[str, Any]], save_path: Path) -> str:
    """Draw green boxes, input: source image/boxes/save path -> output: saved path."""
    image = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, _ = image.size
    line_width = max(3, width // 300)
    for item in boxes:
        box = [int(v) for v in item["box"][:4]]
        draw.rectangle(box, outline=(0, 220, 0), width=line_width)
        label = str(item.get("label") or "").strip()
        if label:
            x1, y1 = box[0], max(0, box[1] - 22)
            draw.rectangle([x1, y1, x1 + 86, y1 + 20], fill=(0, 160, 0))
            draw.text((x1 + 4, y1 + 3), label, fill=(255, 255, 255))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(save_path) + ".tmp.jpg"
    image.save(tmp, format="JPEG", quality=92)
    os.replace(tmp, save_path)
    return str(save_path)


def _draw_boxed_crop(source_path: str, item: Dict[str, Any], save_path: Path) -> str:
    """Crop target region and keep green box, input: source/box/save -> output: saved path."""
    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    box = [float(v) for v in item["box"][:4]]
    bw, bh = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
    x1 = int(max(0, box[0] - 0.30 * bw))
    y1 = int(max(0, box[1] - 0.18 * bh))
    x2 = int(min(width, box[2] + 0.30 * bw))
    y2 = int(min(height, box[3] + 0.12 * bh))
    crop = image.crop((x1, y1, x2, y2))
    rel = [int(box[0] - x1), int(box[1] - y1), int(box[2] - x1), int(box[3] - y1)]
    if crop.size[0] < 900:
        scale = min(2.5, 900 / max(crop.size[0], 1))
        crop = crop.resize((int(crop.size[0] * scale), int(crop.size[1] * scale)), Image.Resampling.LANCZOS)
        rel = [int(v * scale) for v in rel]
    draw = ImageDraw.Draw(crop)
    line_width = max(4, crop.size[0] // 120)
    draw.rectangle(rel, outline=(0, 220, 0), width=line_width)
    draw.rectangle([rel[0], max(0, rel[1] - 24), rel[0] + 96, max(20, rel[1] - 2)], fill=(0, 160, 0))
    draw.text((rel[0] + 4, max(0, rel[1] - 21)), "Target", fill=(255, 255, 255))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(save_path) + ".tmp.jpg"
    crop.save(tmp, format="JPEG", quality=94)
    os.replace(tmp, save_path)
    return str(save_path)


def _copy_selected_frames(selected_infos: List[Dict[str, Any]], save_dir: Path, prefix: str, kind: str, skip_idxs: Optional[set] = None) -> List[Dict[str, Any]]:
    """Copy selected frames to boxed dir, input: infos/save/prefix/kind/skip -> output: copied infos."""
    copied = []
    skip_idxs = skip_idxs or set()
    for idx, info in enumerate(selected_infos, 1):
        if idx in skip_idxs:
            continue
        source = str(info.get("path") or "")
        if not source or not os.path.exists(source):
            continue
        second = info.get("second")
        suffix = Path(source).suffix.lower() or ".jpg"
        out_path = save_dir / f"{prefix}_{idx:03d}_frame_{int(second or 0):04d}s{suffix}"
        save_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + f".{os.getpid()}.tmp")
        shutil.copy2(source, tmp)
        os.replace(tmp, out_path)
        copied.append({
            "path": str(out_path),
            "second": second,
            "source_path": source,
            "kind": kind,
            "boxes": [],
        })
    return copied


def _boxed_candidates(candidates: List[Dict[str, Any]], save_dir: Path) -> List[Dict[str, Any]]:
    """Box final target candidates, input: candidates/save dir -> output: boxed main frames."""
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for rec in candidates:
        path = str(rec.get("path") or "")
        idx = int(rec.get("idx") or 0)
        if not path or not idx:
            continue
        grouped.setdefault((idx, path), []).append(rec)

    boxed = []
    for (idx, path), records in sorted(grouped.items()):
        for box_idx, rec in enumerate(records, 1):
            width, height = rec.get("size") or _image_size(path)
            box = _normalize_box(rec.get("box"), width, height)
            if not box:
                continue
            box_item = {
                "box": box,
                "label": "Target",
                "source": f"selected_{rec.get('evidence') or 'product'}",
                "desc": rec.get("desc"),
                "locator_certainty": rec.get("locator_certainty"),
                "crop_allowed": True,
                "valid": True,
            }
            out_path = save_dir / f"boxed_{idx:03d}_{box_idx:02d}_frame_{int(rec.get('second') or 0):04d}s.jpg"
            boxed.append({
                "path": _draw_boxes(path, [box_item], out_path),
                "second": rec.get("second"),
                "source_path": path,
                "source_idx": idx,
                "kind": "boxed_frame",
                "boxes": [box_item],
                "target_source": "generic_product_grounding",
            })
    return boxed


def _remove_boxed_files(boxed_frames: List[Dict[str, Any]]) -> None:
    """Remove provisional boxed images, input: boxed metadata -> output: none."""
    for frame in boxed_frames:
        path = Path(str(frame.get("path") or ""))
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def _review_boxed_candidates(
    boxed_frames: List[Dict[str, Any]],
    target_text: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Review boxed images one by one, input: boxed/task -> output: final boxes/reviews/pass."""
    reviewed_frames = []
    review_records = []
    for review_index, frame in enumerate(boxed_frames, 1):
        boxes = frame.get("boxes") or []
        box_item = dict(boxes[0]) if boxes and isinstance(boxes[0], dict) else {}
        original_bbox = box_item.get("box")
        try:
            review = review_boxed_frame(frame, target_text)
        except Exception as exc:
            review = {
                "verdict": "reject",
                "corrected_bbox": None,
                "reason": f"reviewer_exception: {str(exc)[:160]}",
                "model": None,
                "raw_response": "",
            }

        verdict = str(review.get("verdict") or "reject").strip().lower()
        final_bbox = original_bbox
        if verdict == "correct":
            source_path = str(frame.get("source_path") or "")
            try:
                width, height = _image_size(source_path)
            except Exception:
                width, height = 0, 0
            final_bbox = _normalize_box(review.get("corrected_bbox"), width, height) if width and height else None
            if not final_bbox:
                verdict = "reject"
                review["reason"] = str(review.get("reason") or "invalid_corrected_bbox")
        elif verdict != "accept":
            verdict = "reject"
            final_bbox = None

        record = {
            "review_index": review_index,
            "boxed_path": frame.get("path"),
            "source_idx": frame.get("source_idx"),
            "original_bbox": original_bbox,
            "verdict": verdict,
            "corrected_bbox": review.get("corrected_bbox") if verdict == "correct" else None,
            "final_bbox": final_bbox,
            "reason": str(review.get("reason") or ""),
            "model": review.get("model"),
            "raw_response": str(review.get("raw_response") or ""),
        }
        review_records.append(record)
        if verdict == "reject":
            return [], review_records, False

        final_frame = dict(frame)
        final_box_item = dict(box_item)
        final_box_item["box"] = final_bbox
        final_box_item["review_verdict"] = verdict
        final_box_item["review_reason"] = record["reason"]
        final_frame["boxes"] = [final_box_item]
        if verdict == "correct":
            source_path = str(final_frame.get("source_path") or "")
            final_path_text = str(final_frame.get("path") or "")
            if not source_path or not final_path_text:
                record["verdict"] = "reject"
                record["reason"] = record["reason"] or "missing_source_or_boxed_path"
                return [], review_records, False
            try:
                _draw_boxes(source_path, [final_box_item], Path(final_path_text))
            except Exception as exc:
                record["verdict"] = "reject"
                record["reason"] = record["reason"] or f"corrected_box_redraw_failed: {str(exc)[:120]}"
                return [], review_records, False
        reviewed_frames.append(final_frame)
    return reviewed_frames, review_records, True


def _append_target_crops(boxed: List[Dict[str, Any]], save_dir: Path) -> List[Dict[str, Any]]:
    """Add boxed zoom crops before full frames, input: boxed frames -> output: augmented frames."""
    out = []
    for frame_idx, frame in enumerate(boxed, 1):
        source = frame.get("source_path")
        if source:
            for box_idx, box_item in enumerate(frame.get("boxes") or [], 1):
                if not box_item.get("box") or box_item.get("crop_allowed") is False:
                    continue
                crop_path = save_dir / f"crop_{frame_idx:03d}_{box_idx:02d}_frame_{int(frame.get('second') or 0):04d}s.jpg"
                out.append({
                    "path": _draw_boxed_crop(source, box_item, crop_path),
                    "second": frame.get("second"),
                    "source_path": source,
                    "kind": "target_crop",
                    "boxes": [box_item],
                })
        out.append(frame)
    return out


def annotate_visual_image_base64_frames(
    selected_infos: List[Dict[str, Any]],
    all_frame_infos: Optional[List[Dict[str, Any]]],
    target_text: str,
    final_scenario: Any,
    final_task_id: Any,
    final_request_seq: Any,
    visual_facts: Optional[Any] = None,
    target_cardinality: Optional[Dict[str, Any]] = None,
    auxiliary_infos: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Box image_base64 frames, input: selected/all frames/target ids -> output: frame infos/meta."""
    if not _scenario_enabled(final_scenario):
        return selected_infos, {"status": "skipped", "reason": "scenario_not_enabled"}
    if not isinstance(target_cardinality, dict) or target_cardinality.get("cardinality") not in {"single", "multiple"}:
        return selected_infos, {"status": "skipped", "reason": "missing_frame_selector_cardinality"}
    try:
        max_targets = int(target_cardinality.get("max_targets"))
    except (TypeError, ValueError):
        return selected_infos, {"status": "skipped", "reason": "missing_frame_selector_cardinality"}
    target_cardinality = {
        "cardinality": target_cardinality["cardinality"],
        "max_targets": 1 if target_cardinality["cardinality"] == "single" else max(1, min(max_targets, 6)),
        "source": "frame_selector",
    }

    request_seq = int(final_request_seq or 1)
    scenario_label = str(final_scenario or "unknown")
    task_label = f"task{int(final_task_id)}" if final_task_id is not None else "task_unknown"
    save_dir = BOX_ROOT / scenario_label / task_label / f"request_{request_seq:03d}"
    if not selected_infos:
        return selected_infos, {"status": "skipped", "reason": "empty_selected_infos"}

    source_dir = _request_source_dir(final_scenario, final_task_id, request_seq)
    selected = _repair_selected_paths([dict(item) for item in selected_infos if item.get("path")], source_dir)
    selected = [item for item in selected if Path(str(item.get("path") or "")).exists()]
    auxiliary = _repair_selected_paths([dict(item) for item in (auxiliary_infos or []) if item.get("path")], source_dir)
    auxiliary = [item for item in auxiliary if Path(str(item.get("path") or "")).exists()]
    if not selected:
        return selected_infos, {"status": "fallback", "reason": "no_existing_selected_frames"}

    signature = _source_signature(selected, auxiliary, target_text, final_scenario, target_cardinality)
    cached_infos, cached_manifest = _load_cached(save_dir, signature)
    if cached_infos:
        return cached_infos, {"status": "boxed", "cached": True, **(cached_manifest or {})}

    save_dir.mkdir(parents=True, exist_ok=True)
    _clear_outputs(save_dir)

    raw_candidates = _locate_selected(selected, target_text, target_cardinality, visual_facts=visual_facts)
    has_nonconfident_candidate = any(
        candidate.get("locator_certainty") != "confident"
        for candidate in raw_candidates
    )
    accepted_candidates = [
        candidate
        for candidate in raw_candidates
        if candidate.get("locator_certainty") == "confident"
    ]
    selected_candidates = (
        []
        if has_nonconfident_candidate
        else _select_final_candidates(accepted_candidates, target_text, target_cardinality)
    )
    boxed_main = _boxed_candidates(selected_candidates, save_dir) if selected_candidates else []
    box_reviews = []
    review_status = "not_run"

    if has_nonconfident_candidate:
        sent_frames = _copy_selected_frames(selected, save_dir, "selected_original", "selected_frame")
        auxiliary_frames = _copy_selected_frames(auxiliary, save_dir, "auxiliary", "auxiliary_frame")
        sent_frames.extend(auxiliary_frames)
        status = "locator_uncertain_fallback_original"
    elif not selected_candidates:
        sent_frames = _copy_selected_frames(selected, save_dir, "selected_original", "selected_frame")
        auxiliary_frames = _copy_selected_frames(auxiliary, save_dir, "auxiliary", "auxiliary_frame")
        sent_frames.extend(auxiliary_frames)
        status = "no_confident_locator_box_fallback_original"
    elif not boxed_main:
        sent_frames = _copy_selected_frames(selected, save_dir, "selected_original", "selected_frame")
        auxiliary_frames = _copy_selected_frames(auxiliary, save_dir, "auxiliary", "auxiliary_frame")
        sent_frames.extend(auxiliary_frames)
        status = "target_box_failed_fallback_original"
    else:
        provisional_boxed = boxed_main
        boxed_main, box_reviews, review_passed = _review_boxed_candidates(provisional_boxed, target_text)
        if not review_passed:
            _remove_boxed_files(provisional_boxed)
            boxed_main = []
            sent_frames = _copy_selected_frames(selected, save_dir, "selected_original", "selected_frame")
            auxiliary_frames = _copy_selected_frames(auxiliary, save_dir, "auxiliary", "auxiliary_frame")
            sent_frames.extend(auxiliary_frames)
            review_status = "rejected"
            status = "box_review_rejected_fallback_original"
        else:
            review_status = "corrected" if any(item.get("verdict") == "correct" for item in box_reviews) else "accepted"
            auxiliary_frames = _copy_selected_frames(auxiliary, save_dir, "auxiliary", "auxiliary_frame")
            sent_frames = _append_target_crops(boxed_main, save_dir) + auxiliary_frames
            status = "boxed"

    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "status": status,
        "method": "generic_product_grounding",
        "signature": signature,
        "target": target_text,
        "scenario": final_scenario,
        "box_filter_mode": "binary_certainty",
        "review_status": review_status,
        "box_reviews": box_reviews,
        "target_cardinality": target_cardinality,
        "raw_candidates": raw_candidates,
        "accepted_candidates": accepted_candidates,
        "selected_candidates": selected_candidates,
        "boxed_main_frames": boxed_main,
        "auxiliary_frames": auxiliary_frames,
        "sent_frames": sent_frames,
        "boxed_frames": sent_frames,
    }
    previous_visual_facts = _normalize_visual_facts_for_prompt(visual_facts)
    if previous_visual_facts:
        manifest["previous_visual_facts"] = previous_visual_facts
    (save_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return [
        {
            "path": item["path"],
            "second": item.get("second"),
            "kind": item.get("kind"),
            "paired_anchor_second": item.get("paired_anchor_second"),
        }
        for item in sent_frames
    ], manifest
