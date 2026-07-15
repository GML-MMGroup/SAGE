"""
脚本作用：
提供多轮交互沙盒所需的媒体消息构造、工具调用执行、工具调用识别和模拟用户回复修正工具函数。

执行逻辑：
1. 判断本地路径或 URL 的媒体类型，并按模型需要构造图像/视频消息。
2. 解析服务智能体输出中的 JSON 工具调用，并在对应数据库实例上执行。
3. 调用用户修正器检查模拟用户回复是否偏离任务，必要时生成修正后的回复。

运行示例：
    该文件由 run/multi_agent.py 导入调用，一般不直接运行。
"""
import os
import sys
import json
import base64
import re
import time
import random
import mimetypes
import subprocess
import shutil

# 将项目根目录加入模块搜索路径，方便从 run 目录直接导入项目模块。
current_file_path = os.path.abspath(__file__)
run_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(run_dir)
sys.path.insert(0, os.path.abspath(project_root))

from config.user_agent_config import USER_CORRECTOR_MODEL_NAME, USER_CORRECTOR_SYSTEM_PROMPT, call_user_corrector_model
from config.frames_selecter_config import (
    FRAME_SELECTER_DEBUG_STDOUT,
    FRAME_SELECTER_FRAME_INTERVAL_SECONDS,
    FRAME_SELECTER_MAX_SELECTED_FRAMES,
    FRAME_SELECTER_MODEL_NAME,
    FRAME_SELECTER_SYSTEM_PROMPT,
    call_frame_selecter_model,
)
from config.visual_boxed_config import VISUAL_BOXED_SCENARIO_PREFIXES
from run import stage_latency
from run.apis import call_llm
from run.prompts import FRAME_SELECTER_PROMPT, USER_CORRECTOR_PROMPT

FRAME_SELECTOR_CONTRACT_VERSION = "frames_cardinality_aux_v2"
_VISUAL_TASK_WRAPPER_RE = re.compile(
    r"^\s*Please\s+identify\s+the\s+visual\s+target\.?\s*Task:\s*",
    re.IGNORECASE,
)


def strip_wrapper(task):
    """Strip the legacy visual-task wrapper, input: task text -> output: clean task text."""
    if not task:
        return task or ""
    return _VISUAL_TASK_WRAPPER_RE.sub("", str(task)).strip()


# --- 媒体处理工具函数 ---


def _visual_box_scenario_enabled(scenario):
    """Check boxed grounding scope, input: scenario -> output: bool."""
    label = str(scenario or "").strip().lower()
    return any(label.startswith(prefix) for prefix in VISUAL_BOXED_SCENARIO_PREFIXES)


def _normalize_visual_facts_for_prompt(visual_facts):
    """Normalize visual facts for prompt context, input: facts -> output: list."""
    if isinstance(visual_facts, list):
        return [fact for fact in visual_facts if isinstance(fact, dict)]
    if isinstance(visual_facts, dict):
        return [visual_facts] if visual_facts else []
    return []


def _format_previous_visual_facts_context(visual_facts):
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


def encode_image(image_path):
    """将图片文件转为base64字符串，输入：图片路径 -> 输出：base64字符串或None"""
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


"""读取视频时长，输入：视频路径->输出：视频秒数"""
def get_video_duration(video_path):
    if shutil.which("ffprobe"):
        return float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            text=True
        ).strip())

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("ffprobe is unavailable and OpenCV could not open video: " + str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError("Unable to determine video duration without ffprobe: " + str(video_path))
    return float(frame_count / fps)


"""保存指定时间戳视频帧，输入：视频路径、秒数、保存路径->输出：无"""
def save_video_frame(video_path, timestamp, frame_path):
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path, "-frames:v", "1", "-q:v", "2", frame_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("ffmpeg is unavailable and OpenCV could not open video: " + str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("OpenCV failed to read video frame at second " + str(timestamp))
    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
    cv2.imwrite(frame_path, frame)


"""按固定秒间隔从视频抽帧，输入：视频路径、秒间隔->输出：包含路径和秒数的帧信息列表"""
def extract_video_frames_by_seconds(video_path, frame_interval_seconds=1):
    interval = max(1, int(float(frame_interval_seconds or 1)))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(project_root, "processed", "frames", f"frame_{interval}s", video_name)
    os.makedirs(save_dir, exist_ok=True)

    cached_frames = []
    for image_name in sorted(os.listdir(save_dir)):
        match = re.match(r"^frame_(\d+)s\.(jpg|jpeg|png|webp)$", image_name, re.IGNORECASE)
        if match:
            cached_frames.append({"path": os.path.join(save_dir, image_name), "second": int(match.group(1))})

    try:
        duration = get_video_duration(video_path)
    except Exception:
        if cached_frames:
            return cached_frames
        raise

    seconds = list(range(0, max(0, int(duration)) + 1, interval)) or [0]
    frame_infos = []
    for second in seconds:
        frame_path = os.path.join(save_dir, f"frame_{second:04d}s.jpg")
        if not os.path.exists(frame_path):
            timestamp = min(float(second), max(0.0, duration - 0.05))
            save_video_frame(video_path, timestamp, frame_path)
        if os.path.exists(frame_path):
            frame_infos.append({"path": frame_path, "second": second})
    return frame_infos


"""解析模型输出中的JSON对象，输入：模型文本->输出：dict或None"""
def _extract_json_dict(text):
    text = str(text or "").strip()
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


"""获取image_base64选帧缓存目录，输入：场景、任务id、请求序号->输出：目录路径"""
def _frame_select_cache_dir(scenario, task_id, request_seq):
    scenario = str(scenario or "unknown")
    task_id = f"task{task_id}" if task_id is not None else "task_unknown"
    request_seq = int(request_seq or 1)
    return os.path.join(project_root, "processed", "frame_select", scenario, task_id, f"request_{request_seq:03d}")


"""从文件名解析帧秒数，输入：图片路径->输出：秒数或None"""
def _parse_frame_second(image_path):
    match = re.search(r"frame_(\d+)s", os.path.basename(str(image_path)))
    if not match:
        return None
    return int(match.group(1))


"""读取已缓存的选帧，输入：缓存目录->输出：帧信息列表"""
def _load_cached_selected_frames(cache_dir):
    selected = []
    for image_name in sorted(os.listdir(cache_dir)):
        if not image_name.startswith("selected_") or not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        image_path = os.path.join(cache_dir, image_name)
        selected.append({"path": image_path, "second": _parse_frame_second(image_path)})
    return selected


def _load_cached_auxiliary_frames(cache_dir):
    """Load selector-provided auxiliary frames from cache."""
    auxiliary = []
    for image_name in sorted(os.listdir(cache_dir)):
        if not image_name.startswith("auxiliary_") or not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        image_path = os.path.join(cache_dir, image_name)
        auxiliary.append({
            "path": image_path,
            "second": _parse_frame_second(image_path),
            "kind": "auxiliary_frame",
        })
    return auxiliary


"""规范化选帧器数量判断，输入：响应字典->输出：数量判断或None"""
def _normalize_frame_selector_cardinality(data):
    if not isinstance(data, dict):
        return None
    cardinality = str(data.get("cardinality") or "").strip().lower()
    if cardinality not in {"single", "multiple"}:
        return None
    if cardinality == "single":
        return {"cardinality": "single", "max_targets": 1, "source": "frame_selector"}
    raw_max_targets = data.get("max_targets")
    if isinstance(raw_max_targets, bool):
        return None
    try:
        numeric_max_targets = float(raw_max_targets)
    except (TypeError, ValueError):
        return None
    if not numeric_max_targets.is_integer():
        return None
    max_targets = int(numeric_max_targets)
    return {
        "cardinality": "multiple",
        "max_targets": max(1, min(max_targets, 6)),
        "source": "frame_selector",
    }


"""读取带数量判断的选帧缓存，输入：缓存目录->输出：帧/数量/缓存版本是否有效"""
def _load_cached_frame_selection(cache_dir):
    manifest_path = os.path.join(cache_dir, "manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return [], [], None, False
    if manifest.get("frame_selector_contract_version") != FRAME_SELECTOR_CONTRACT_VERSION:
        return [], [], None, False
    return (
        _load_cached_selected_frames(cache_dir),
        _load_cached_auxiliary_frames(cache_dir),
        _normalize_frame_selector_cardinality(manifest.get("target_cardinality")),
        True,
    )


"""构造selecter消息，输入：任务文本、帧信息、图片detail->输出：messages"""
def _build_frame_selecter_messages(task_text, frame_infos, image_detail=None, visual_facts=None):
    clean_task_text = strip_wrapper(task_text)
    user_prompt = f"Task: {clean_task_text}"
    previous_visual_facts_context = _format_previous_visual_facts_context(visual_facts)
    if previous_visual_facts_context:
        user_prompt = f"{user_prompt}\n\n{previous_visual_facts_context}"
    if FRAME_SELECTER_PROMPT.strip():
        user_prompt = f"{user_prompt}\n\n{FRAME_SELECTER_PROMPT.strip()}"
    content = [{"type": "text", "text": user_prompt}]
    content.append({"type": "text", "text": "The temporal information for each frame is listed below."})
    for idx, frame_info in enumerate(frame_infos, 1):
        content.append({"type": "text", "text": f"Frame {idx}: second {frame_info.get('second')}."})
        base64_media = encode_image(frame_info.get("path"))
        if base64_media:
            content.append(_build_image_content(f"data:image/jpeg;base64,{base64_media}", image_detail))
    return [
        {"role": "system", "content": FRAME_SELECTER_SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]


"""解析selecter返回的帧与数量，输入：模型文本、最大编号->输出：合法索引和数量判断"""
def _parse_frame_selection(response_text, max_index):
    data = _extract_json_dict(response_text)
    if not data:
        return [], [], None
    frames = data.get("frames", [])
    if not isinstance(frames, list):
        return [], [], _normalize_frame_selector_cardinality(data)

    indexes = []
    for item in frames:
        if isinstance(item, dict):
            item = item.get("frame") or item.get("index") or item.get("id")
        try:
            frame_idx = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= frame_idx <= max_index and frame_idx not in indexes:
            indexes.append(frame_idx)
        if len(indexes) >= FRAME_SELECTER_MAX_SELECTED_FRAMES:
            break
    auxiliary_indexes = []
    for item in data.get("auxiliary_frames", []) if isinstance(data.get("auxiliary_frames", []), list) else []:
        if isinstance(item, dict):
            item = item.get("frame") or item.get("index") or item.get("id")
        try:
            frame_idx = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= frame_idx <= max_index and frame_idx not in indexes and frame_idx not in auxiliary_indexes:
            auxiliary_indexes.append(frame_idx)
        if len(auxiliary_indexes) >= FRAME_SELECTER_MAX_SELECTED_FRAMES:
            break
    return indexes, auxiliary_indexes, _normalize_frame_selector_cardinality(data)


"""解析selecter返回的帧编号，输入：模型文本、最大编号->输出：合法1-based编号列表"""
def _parse_selected_frame_indexes(response_text, max_index):
    indexes, _, _ = _parse_frame_selection(response_text, max_index)
    return indexes


"""Keep the selector response unchanged in normal and debug manifests."""
def _frame_select_manifest_response(response_text, selected_indexes):
    return response_text


"""保存selecter选中的帧，输入：选中帧信息、缓存目录、元数据->输出：保存后的帧信息"""
def _save_selected_frames(selected_infos, auxiliary_infos, cache_dir, manifest):
    os.makedirs(cache_dir, exist_ok=True)
    for image_name in os.listdir(cache_dir):
        if (image_name.startswith("selected_") or image_name.startswith("auxiliary_")) and image_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            os.remove(os.path.join(cache_dir, image_name))
    saved_infos = []
    for idx, frame_info in enumerate(selected_infos, 1):
        second = frame_info.get("second")
        save_name = f"selected_{idx:03d}_frame_{int(second):04d}s.jpg"
        save_path = os.path.join(cache_dir, save_name)
        shutil.copy2(frame_info["path"], save_path)
        saved_infos.append({"path": save_path, "second": second})
    manifest["selected_frames"] = [
        {"path": item["path"], "second": item.get("second")} for item in saved_infos
    ]
    saved_auxiliary = []
    for idx, frame_info in enumerate(auxiliary_infos or [], 1):
        second = frame_info.get("second")
        save_name = f"auxiliary_{idx:03d}_frame_{int(second):04d}s.jpg"
        save_path = os.path.join(cache_dir, save_name)
        shutil.copy2(frame_info["path"], save_path)
        saved_auxiliary.append({"path": save_path, "second": second, "kind": "auxiliary_frame"})
    manifest["auxiliary_frames"] = [
        {"path": item["path"], "second": item.get("second")} for item in saved_auxiliary
    ]
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return saved_infos, saved_auxiliary


"""构造image_base64选帧说明，输入：boxed元数据->输出：说明文本"""
def _frame_selected_intro(box_meta=None):
    if isinstance(box_meta, dict) and box_meta.get("status") == "boxed":
        return "The following images have been processed by frame selection and target marking to help identify the item."
    return "The following frames were selected as relevant to the current visual recognition task."


"""为image_base64选帧加框，输入：选中帧/全部帧/目标/数量/场景task/开关->输出：帧信息和元数据"""
def _maybe_box_selected_frames(
    selected_infos,
    auxiliary_infos,
    frame_infos,
    task_text,
    scenario,
    task_id,
    request_seq,
    box_enabled=True,
    visual_facts=None,
    target_cardinality=None,
):
    if not box_enabled:
        return selected_infos, None
    if not _visual_box_scenario_enabled(scenario):
        return selected_infos, None
    if not target_cardinality:
        return selected_infos, {"status": "skipped", "reason": "missing_frame_selector_cardinality"}
    profile = stage_latency.start("frame_boxer", model="visual_boxed_frames")
    try:
        from run.visual_boxed_frames import annotate_visual_image_base64_frames
        boxed_infos, box_meta = annotate_visual_image_base64_frames(
            selected_infos=selected_infos,
            auxiliary_infos=auxiliary_infos,
            all_frame_infos=frame_infos,
            target_text=task_text,
            final_scenario=scenario,
            final_task_id=task_id,
            final_request_seq=request_seq,
            visual_facts=visual_facts,
            target_cardinality=target_cardinality,
        )
        stage_latency.end(
            profile,
            cached=bool(isinstance(box_meta, dict) and box_meta.get("cached"))
        )
        return boxed_infos, box_meta
    except Exception as e:
        stage_latency.end(profile, status="error")
        print(f"[visual_box] failed, fallback to original selected frames: {str(e)[:160]}", flush=True)
        return selected_infos, {"status": "error", "message": str(e)}


"""向content追加帧和时间，输入：content、帧信息、图片detail->输出：content"""
def _append_frame_image_content(content, frame_infos, image_detail=None):
    for idx, frame_info in enumerate(frame_infos, 1):
        second = frame_info.get("second")
        second_text = "unknown" if second is None else str(second)
        image_name = os.path.basename(str(frame_info.get("path") or "")) or "unknown"
        if frame_info.get("kind") == "target_crop":
            text = (
                f"Image {idx}: file {image_name}, zoomed crop from second {second_text}. "
                "This crop shows the likely target anchor marked by a green Target box. The box may enclose the physical target, "
                "the text target itself, or physically associated text and object evidence. Combine physical and textual evidence only "
                "when visible evidence confirms that they refer to the same entity."
            )
        elif frame_info.get("kind") == "boxed_frame":
            text = (
                f"Image {idx}: file {image_name}, boxed selected frame at second {second_text}. "
                "The green Target box marks the likely target anchor and may cover a physical item, a text target, or their associated "
                "evidence. Prefer it only when it is consistent with the task wording and visible context."
            )
        elif frame_info.get("kind") == "auxiliary_frame":
            text = (
                f"Image {idx}: file {image_name}, unboxed auxiliary selected frame at second {second_text}. "
                "It may more clearly show the same target's physical item or textual representation, such as its name, label, price, "
                "menu item, or order entry. Combine that text with a physical item only when same-frame association, position/layout, "
                "temporal continuity, or explicit pairing confirms the same entity; do not borrow text from a nearby item, another row/page, or another menu."
            )
        elif frame_info.get("kind") in {"clean_context_crop", "clean_context_boxed"}:
            anchor_second = frame_info.get("paired_anchor_second")
            anchor_text = "the boxed target frame" if anchor_second is None else f"the boxed target at second {anchor_second}"
            text = (
                f"Image {idx}: file {image_name}, clean reprojected context at second {second_text}, paired with {anchor_text}. "
                "The green Target box has been geometrically relocated to the same target in this cleaner frame; use it to read unobstructed label, menu text, or appearance details."
            )
        elif frame_info.get("kind") == "clean_context":
            anchor_second = frame_info.get("paired_anchor_second")
            anchor_text = "the boxed target frame" if anchor_second is None else f"the boxed target at second {anchor_second}"
            text = (
                f"Image {idx}: file {image_name}, clean adjacent context frame at second {second_text}, paired with {anchor_text}. "
                "Use this image to read unobstructed label, menu text, or appearance details for the likely boxed target."
            )
        else:
            text = (
                f"Image {idx}: file {image_name}, selected frame at second {second_text}. "
                "If a green Target box is present, treat it as the likely target anchor and verify it against the task wording and visible context. "
                "When combining physical and textual evidence across selected frames, first confirm that they refer to the same entity."
            )
        content.append({"type": "text", "text": text})
        base64_media = encode_image(frame_info.get("path"))
        if base64_media:
            image_ref = f"data:{get_media_mime_type(frame_info.get('path'))};base64,{base64_media}"
            content.append(_build_image_content(image_ref, image_detail))
    return content


"""为image_base64构造选帧后的图片输入，输入：文本、视频、场景/task/request->输出：content"""
def _build_frame_selected_base64_content(text, media_path, image_detail=None, scenario=None, target=None, task_id=None, request_seq=None, box_enabled=True, visual_facts=None):
    content = [
        {"type": "text", "text": text}
    ]
    cache_dir = _frame_select_cache_dir(scenario, task_id, request_seq)
    if os.path.isdir(cache_dir):
        selector_wall = stage_latency.start_wall("frame_selector", model="frame_selector_cache", cached=True)
        try:
            selected_infos, auxiliary_infos, target_cardinality, cache_valid = _load_cached_frame_selection(cache_dir)
        except Exception:
            stage_latency.end_wall(selector_wall, status="error", cached=True)
            raise
        if cache_valid and selected_infos:
            cache_profile = stage_latency.start("frame_selector", model=FRAME_SELECTER_MODEL_NAME, cached=True)
            stage_latency.end(cache_profile, cached=True)
            stage_latency.end_wall(selector_wall, cached=True)
            selected_infos, box_meta = _maybe_box_selected_frames(
                selected_infos, auxiliary_infos, None, target or text, scenario, task_id, request_seq,
                box_enabled=box_enabled, visual_facts=visual_facts,
                target_cardinality=target_cardinality,
            )
            content.append({"type": "text", "text": _frame_selected_intro(box_meta)})
            return _append_frame_image_content(content, selected_infos, image_detail)
        stage_latency.end_wall(selector_wall, cached=True)
        if cache_valid:
            print(f"[frame_selecter] current cache has no selected frames, regenerating: {cache_dir}", flush=True)
        else:
            print(f"[frame_selecter] legacy or invalid cache contract, regenerating: {cache_dir}", flush=True)

    extract_wall = stage_latency.start_wall("frame_selector", model="frame_extraction")
    try:
        frame_infos = extract_video_frames_by_seconds(media_path, FRAME_SELECTER_FRAME_INTERVAL_SECONDS)
    except Exception:
        stage_latency.end_wall(extract_wall, status="error")
        raise
    else:
        stage_latency.end_wall(extract_wall)
    if not frame_infos:
        content.append({"type": "text", "text": _frame_selected_intro(None)})
        return content

    task_text = target or text
    messages = _build_frame_selecter_messages(task_text, frame_infos, image_detail, visual_facts=visual_facts)
    selector_profile = stage_latency.start("frame_selector", model=FRAME_SELECTER_MODEL_NAME)
    try:
        response, input_tokens, output_tokens = call_frame_selecter_model(messages)
    except Exception:
        stage_latency.end(selector_profile, status="error")
        raise
    else:
        stage_latency.end(selector_profile, input_tokens, output_tokens)
    if FRAME_SELECTER_DEBUG_STDOUT:
        print(f"[frame_selecter debug raw response] {str(response).strip()}", flush=True)
    selected_indexes, auxiliary_indexes, target_cardinality = _parse_frame_selection(response, len(frame_infos))
    if not selected_indexes:
        print(f"[frame_selecter] no valid selected frames, fallback to all sampled frames: {str(response)[:160]}", flush=True)
        return _append_frame_image_content(content, frame_infos, image_detail)

    selected_infos = [frame_infos[idx - 1] for idx in selected_indexes]
    auxiliary_infos = [frame_infos[idx - 1] for idx in auxiliary_indexes]
    manifest = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "scenario": scenario,
        "task_id": task_id,
        "request_seq": request_seq,
        "media_path": media_path,
        "target": target,
        "frame_interval_seconds": FRAME_SELECTER_FRAME_INTERVAL_SECONDS,
        "response": _frame_select_manifest_response(response, selected_indexes),
        "frame_selector_contract_version": FRAME_SELECTOR_CONTRACT_VERSION,
        "target_cardinality": target_cardinality,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    previous_visual_facts = _normalize_visual_facts_for_prompt(visual_facts)
    if previous_visual_facts:
        manifest["previous_visual_facts"] = previous_visual_facts
    save_wall = stage_latency.start_wall("frame_selector", model="frame_select_save")
    try:
        selected_infos, auxiliary_infos = _save_selected_frames(selected_infos, auxiliary_infos, cache_dir, manifest)
    except Exception:
        stage_latency.end_wall(save_wall, status="error")
        raise
    else:
        stage_latency.end_wall(save_wall)
    selected_infos, box_meta = _maybe_box_selected_frames(
        selected_infos, auxiliary_infos, frame_infos, task_text, scenario, task_id, request_seq,
        box_enabled=box_enabled, visual_facts=visual_facts,
        target_cardinality=target_cardinality,
    )
    content.append({"type": "text", "text": _frame_selected_intro(box_meta)})
    return _append_frame_image_content(content, selected_infos, image_detail)


def get_media_mime_type(file_path):
    """根据本地路径获取MIME类型，输入：图片路径 -> 输出：MIME类型字符串"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_map.get(ext, 'image/jpeg')


def _build_image_content(image_ref, image_detail=None):
    """构造图片消息，输入：图片引用/detail -> 输出：OpenAI兼容图片content"""
    payload = {"url": image_ref}
    detail = str(image_detail or "").strip().lower()
    if detail and detail != "none":
        payload["detail"] = detail
    return {"type": "image_url", "image_url": payload}


def build_message_with_media(text, media_path=None, use_vision=False, service_model_name="qwen3-vl-225b", image_detail=None, scenario=None, target=None, task_id=None, request_seq=None, box_enabled=True, visual_facts=None):
    """构造image_base64视觉消息，输入：文本、视频路径、场景/task/request -> 输出：content列表"""
    if use_vision and media_path:
        return _build_frame_selected_base64_content(
            text,
            media_path,
            image_detail=image_detail,
            scenario=scenario,
            target=target,
            task_id=task_id,
            request_seq=request_seq,
            box_enabled=box_enabled,
            visual_facts=visual_facts,
        )
    return [{"type": "text", "text": text}]


def build_message_with_image(text, image_path=None, use_vision=False, service_model_name="qwen3-vl-225b", image_detail=None):
    """兼容旧函数名并转调image_base64消息构造函数，输入：文本/媒体路径 -> 输出：content列表"""
    return build_message_with_media(text, image_path, use_vision, service_model_name, image_detail=image_detail)


# --- 工具执行函数 ---

def execute_tool(db_instance, tool_calls_data):
    """解析并执行工具调用，输入：数据库实例、工具调用字典或列表 -> 输出：结构化工具执行结果列表"""
    if isinstance(tool_calls_data, dict):
        tool_calls_data = [tool_calls_data]
    elif not isinstance(tool_calls_data, list):
        return [{"role": "tool", "tool_name": "unknown", "parameters": {}, "content": json.dumps({"error": "Invalid tool call format. Expected dict or list."}, ensure_ascii=False)}]

    results = []
    for tool_call_obj in tool_calls_data:
        try:
            method_name = tool_call_obj.get("tool_name") or tool_call_obj.get("name")
            if not method_name:
                results.append({
                    "role": "tool",
                    "tool_name": "unknown",
                    "parameters": {},
                    "content": json.dumps({"error": "Missing tool identifier ('tool_name' or 'name')"}, ensure_ascii=False)
                })
                continue

            params = tool_call_obj.get("parameters", tool_call_obj.get("arguments", {}))

            print(f"  [Tool Execution] Calling: {method_name} Parameters: {params}")

            if hasattr(db_instance, method_name):
                method = getattr(db_instance, method_name)
                tool_wall = stage_latency.start_wall("executor", model=method_name)
                try:
                    result = method(**params)
                except Exception:
                    stage_latency.end_wall(tool_wall, status="error")
                    raise
                else:
                    stage_latency.end_wall(tool_wall)
                print(f"  [Tool Execution] Return result: {result}")
                results.append({
                    "role": "tool",
                    "tool_name": method_name,
                    "parameters": params,
                    "content": json.dumps(result, ensure_ascii=False, default=str)
                })
            else:
                results.append({
                    "role": "tool",
                    "tool_name": method_name,
                    "parameters": params,
                    "content": json.dumps({"error": f"Tool '{method_name}' not found"}, ensure_ascii=False)
                })
        except Exception as e:
            results.append({
                "role": "tool",
                "tool_name": tool_call_obj.get("tool_name") or tool_call_obj.get("name") or "unknown",
                "parameters": tool_call_obj.get("parameters", tool_call_obj.get("arguments", {})),
                "content": json.dumps({"error": str(e)}, ensure_ascii=False)
            })

    return results


# --- 工具调用识别函数 ---

def check_tool_call(response_text):
    """从模型回复中提取工具调用，输入：回复文本 -> 输出：(是否为工具调用, 工具调用列表或None)"""
    text = response_text.strip()

    # 统一识别条件，兼容 tool_call、tool_name 和 name 三种字段。
    def is_tool_call(obj):
        """判断对象是否为工具调用，输入：JSON对象 -> 输出：布尔值"""
        return isinstance(obj, dict) and ("tool_call" in obj or "tool_name" in obj or "name" in obj)

    # 优先尝试把整段回复当作JSON解析。
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            tool_calls = [item for item in data if is_tool_call(item)]
            if tool_calls:
                return True, tool_calls
        elif is_tool_call(data):
            return True, [data]
    except json.JSONDecodeError:
        pass

    # 如果整段解析失败，再用正则提取可能的JSON对象片段。
    json_pattern = r'\{([^{}]|(\{[^{}]*\}))*\}'
    potential_jsons = re.findall(json_pattern, text, re.DOTALL)

    valid_tool_calls = []
    for match in potential_jsons:
        json_str = match[0] if isinstance(match, tuple) else match
        if not json_str.startswith('{'):
            json_str = '{' + json_str
        if not json_str.endswith('}'):
            json_str = json_str + '}'

        try:
            obj = json.loads(json_str)
            if is_tool_call(obj):
                valid_tool_calls.append(obj)
        except (json.JSONDecodeError, ValueError):
            continue

    # 兼容Kimi工具微调格式：<|tool_call_begin|>functions.xxx:0<|tool_call_argument_begin|>{...}<|tool_call_end|>。
    if not valid_tool_calls and "<|tool_call_begin|>" in text:
        kimi_pattern = r'<\|tool_call_begin\|>(?:functions\.)?([A-Za-z_][A-Za-z0-9_]*)(?::\d+)?<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>'
        for tool_name, args_str in re.findall(kimi_pattern, text, re.DOTALL):
            try:
                params = json.loads(args_str.strip())
                if isinstance(params, dict):
                    valid_tool_calls.append({"tool_name": tool_name, "parameters": params})
            except (json.JSONDecodeError, ValueError):
                continue

    # 额外兼容JSON数组形式的工具调用。
    if not valid_tool_calls:
        array_pattern = r'\[.*\]'
        array_matches = re.findall(array_pattern, text, re.DOTALL)

        for arr_str in array_matches:
            try:
                arr = json.loads(arr_str)
                if isinstance(arr, list):
                    for item in arr:
                        if is_tool_call(item):
                            valid_tool_calls.append(item)
            except (json.JSONDecodeError, ValueError):
                continue

    if valid_tool_calls:
        return True, valid_tool_calls
    return False, None

# --- 模拟用户回复修正函数 ---

def _extract_json_object(text):
    """提取JSON对象，输入：模型文本 -> 输出：字典或None"""
    text = str(text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _extract_corrector_result(text):
    """提取修正器结果，输入：模型文本 -> 输出：包含修正字段的字典或None"""
    result = _extract_json_object(text)
    if result:
        return result

    text = str(text).strip()
    applied_match = re.search(r'"correction_applied"\s*:\s*(true|false)', text, re.IGNORECASE)
    if not applied_match:
        return None

    def extract_string_field(field_name):
        pattern = rf'"{field_name}"\s*:\s*"([\s\S]*?)"\s*(?:,\s*"[^"]+"\s*:|\s*\}})'
        match = re.search(pattern, text)
        if not match:
            return ""
        value = match.group(1)
        return value.replace('\\"', '"').replace("\\n", "\n").strip()

    return {
        "correction_applied": applied_match.group(1).lower() == "true",
        "correction_reason": extract_string_field("correction_reason"),
        "corrected_response": extract_string_field("corrected_response")
    }


def _format_dialogue_history(dialogue):
    """格式化历史对话，输入：dialogue列表 -> 输出：文本历史"""
    if not dialogue:
        return "None"
    lines = []
    for item in dialogue:
        role = "User" if item.get("role") == "user" else "Service Agent"
        lines.append(f"{role}: {item.get('content', '')}")
    return "\n".join(lines) if lines else "None"


def correct_user_response(user_response, user_instruction, dialogue=None, last_agent_response=""):
    """修正模拟用户回复，输入：用户回复、任务、历史对话、上一轮服务回复 -> 输出：(最终回复, evaluation字典)"""
    prompt = USER_CORRECTOR_PROMPT.format(
        user_instruction=user_instruction,
        dialogue=_format_dialogue_history(dialogue),
        agent_response=last_agent_response,
        user_response=user_response
    )
    messages = [
        {"role": "system", "content": USER_CORRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    print("Correcting user response...")
    corrector_profile = stage_latency.start("user_corrector", model=USER_CORRECTOR_MODEL_NAME)
    try:
        response_text, corrector_input_tokens, corrector_output_tokens = call_user_corrector_model(messages)
    except Exception:
        stage_latency.end(corrector_profile, status="error")
        raise
    else:
        stage_latency.end(corrector_profile, corrector_input_tokens, corrector_output_tokens)
    result = _extract_corrector_result(response_text)
    if not result:
        reason = "用户修正器返回无法解析的JSON，保留原始回复。"
        print(f"User correction failed: {reason}")
        return user_response, {
            "correction_applied": False,
            "correction_reason": reason,
            "raw_response": user_response
        }

    correction_applied = bool(result.get("correction_applied", False))
    correction_reason = str(result.get("correction_reason", "")).strip()
    corrected_response = str(result.get("corrected_response", "")).strip()

    if correction_applied and corrected_response:
        print(f"User Response Corrected: {corrected_response}")
        print(f"Correction Reason: {correction_reason}")
        return corrected_response, {
            "correction_applied": True,
            "correction_reason": correction_reason,
            "raw_response": user_response
        }

    if correction_applied and not corrected_response:
        correction_reason = correction_reason or "用户修正器要求修正但未返回 corrected_response，保留原始回复。"
        print(f"User correction skipped: {correction_reason}")
        return user_response, {
            "correction_applied": False,
            "correction_reason": correction_reason,
            "raw_response": user_response
        }

    return user_response, {
        "correction_applied": False,
        "correction_reason": correction_reason,
        "raw_response": user_response
    }
