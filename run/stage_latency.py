"""Lightweight per-stage latency and token profiling.

This module is intentionally independent from the agent implementation. Callers
only set context, start a stage, and end it with token counts.
"""
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_ROOT = PROJECT_ROOT / "processed" / "profiling"
OFFICIAL_AGENT_STAGES = [
    "frame_selector",
    "frame_boxer",
    "frames_reviewer",
    "visual_recognition",
    "supervisor",
    "planner",
    "executor",
    "reporter",
]
NON_OFFICIAL_USER_STAGES = [
    "user_agent",
    "user_corrector",
]
KNOWN_STAGES = OFFICIAL_AGENT_STAGES + NON_OFFICIAL_USER_STAGES
UNATTRIBUTED_OVERHEAD_STAGE = "unattributed_overhead"
KNOWN_WALL_STAGES = KNOWN_STAGES + [UNATTRIBUTED_OVERHEAD_STAGE]
_enabled = False
_run_id = ""
_profile_dir: Optional[Path] = None
_calls_path: Optional[Path] = None
_wall_events_path: Optional[Path] = None
_task_summaries_path: Optional[Path] = None
_context: Dict[str, Any] = {
    "run_id": None,
    "scenario": None,
    "task_id": None,
    "turn": None,
}
_active_handles = []
_active_wall_handles = []
_warned_write_failure = False


def parse_bool(value: Any, default: bool = True) -> bool:
    """Parse bool-like values, input: value/default -> output: bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def configure(
    enabled: bool = True,
    run_id: Optional[str] = None,
    profile_root: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> None:
    """Configure profiler, input: enabled/run id/root/directory name -> output: none."""
    global _enabled, _run_id, _profile_dir, _calls_path, _wall_events_path, _task_summaries_path
    global _warned_write_failure
    _active_handles.clear()
    _active_wall_handles.clear()
    _warned_write_failure = False
    _enabled = bool(enabled)
    _run_id = str(run_id or time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    _context["run_id"] = _run_id

    if not _enabled:
        _profile_dir = None
        _calls_path = None
        _wall_events_path = None
        _task_summaries_path = None
        return

    root = Path(profile_root) if profile_root else DEFAULT_PROFILE_ROOT
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    _profile_dir = root / str(profile_name or _run_id)
    _calls_path = _profile_dir / "calls.jsonl"
    _wall_events_path = _profile_dir / "wall_events.jsonl"
    _task_summaries_path = _profile_dir / "task_summaries.jsonl"
    _safe_mkdir(_profile_dir)


def set_context(scenario: Any = None, task_id: Any = None, turn: Any = None, run_id: Any = None) -> None:
    """Set current profiling context, input: optional fields -> output: none."""
    if run_id is not None:
        _context["run_id"] = run_id
    if scenario is not None:
        _context["scenario"] = scenario
    if task_id is not None:
        _context["task_id"] = task_id
    _context["turn"] = turn


def start(stage: str, model: Optional[str] = None, cached: bool = False) -> Optional[Dict[str, Any]]:
    """Start a timed stage, input: stage/model/cache flag -> output: handle."""
    if not _enabled:
        return None
    parent_handle = _active_handles[-1] if _active_handles else None
    handle = {
        "stage": str(stage),
        "model": model,
        "cached": bool(cached),
        "start_time": time.perf_counter(),
        "context": dict(_context),
        "acc_input_tokens": 0,
        "acc_output_tokens": 0,
        "child_wall_seconds": 0.0,
        "parent_handle": parent_handle,
    }
    _active_handles.append(handle)
    return handle


def start_wall(stage: str, model: Optional[str] = None, cached: bool = False) -> Optional[Dict[str, Any]]:
    """Start a wall-clock stage, input: stage/model/cache flag -> output: handle."""
    if not _enabled:
        return None
    handle = {
        "stage": str(stage),
        "model": model,
        "cached": bool(cached),
        "start_time": time.perf_counter(),
        "context": dict(_context),
    }
    _active_wall_handles.append(handle)
    return handle


def add_usage(stage: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
    """Add token usage to the latest active stage, input: stage/tokens -> output: none."""
    if not _enabled:
        return
    for handle in reversed(_active_handles):
        if handle.get("stage") == stage:
            handle["acc_input_tokens"] = int(handle.get("acc_input_tokens", 0) or 0) + _to_int(input_tokens)
            handle["acc_output_tokens"] = int(handle.get("acc_output_tokens", 0) or 0) + _to_int(output_tokens)
            return


def end(
    handle: Optional[Dict[str, Any]],
    input_tokens: int = 0,
    output_tokens: int = 0,
    status: str = "success",
    cached: Optional[bool] = None,
) -> None:
    """End a timed stage and write one record, input: handle/tokens/status -> output: none."""
    if not _enabled or not handle:
        return
    end_time = time.perf_counter()
    try:
        start_time = float(handle.get("start_time", end_time))
    except (TypeError, ValueError):
        start_time = end_time
    inclusive_latency = max(0.0, end_time - start_time)
    child_wall_seconds = max(0.0, float(handle.get("child_wall_seconds", 0.0) or 0.0))
    latency = max(0.0, inclusive_latency - child_wall_seconds)
    for idx, active_handle in enumerate(list(_active_handles)):
        if active_handle is handle:
            _active_handles.pop(idx)
            break

    parent_handle = handle.get("parent_handle")
    if isinstance(parent_handle, dict) and any(active is parent_handle for active in _active_handles):
        parent_handle["child_wall_seconds"] = (
            float(parent_handle.get("child_wall_seconds", 0.0) or 0.0) + inclusive_latency
        )

    total_input_tokens = _to_int(input_tokens) + _to_int(handle.get("acc_input_tokens", 0))
    total_output_tokens = _to_int(output_tokens) + _to_int(handle.get("acc_output_tokens", 0))
    record(
        stage=str(handle.get("stage") or "unknown"),
        model=handle.get("model"),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        latency_seconds=latency,
        cached=bool(handle.get("cached") if cached is None else cached),
        status=status,
        context=handle.get("context") if isinstance(handle.get("context"), dict) else None,
    )
    record_wall(
        stage=str(handle.get("stage") or "unknown"),
        model=handle.get("model"),
        wall_seconds=latency,
        cached=bool(handle.get("cached") if cached is None else cached),
        status=status,
        context=handle.get("context") if isinstance(handle.get("context"), dict) else None,
    )


def end_wall(
    handle: Optional[Dict[str, Any]],
    status: str = "success",
    cached: Optional[bool] = None,
) -> None:
    """End a wall-clock stage and write one record, input: handle/status -> output: none."""
    if not _enabled or not handle:
        return
    wall_seconds = time.perf_counter() - float(handle.get("start_time", time.perf_counter()))
    for idx, active_handle in enumerate(list(_active_wall_handles)):
        if active_handle is handle:
            _active_wall_handles.pop(idx)
            break
    record_wall(
        stage=str(handle.get("stage") or "unknown"),
        model=handle.get("model"),
        wall_seconds=wall_seconds,
        cached=bool(handle.get("cached") if cached is None else cached),
        status=status,
        context=handle.get("context") if isinstance(handle.get("context"), dict) else None,
    )


def record(
    stage: str,
    model: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_seconds: float = 0.0,
    cached: bool = False,
    status: str = "success",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one profiling row, input: stage stats -> output: none."""
    if not _enabled:
        return
    ctx = dict(context or _context)
    row = {
        "run_id": ctx.get("run_id") or _run_id,
        "scenario": ctx.get("scenario"),
        "task_id": ctx.get("task_id"),
        "turn": ctx.get("turn"),
        "stage": str(stage),
        "model": model,
        "input_tokens": _to_int(input_tokens),
        "output_tokens": _to_int(output_tokens),
        "latency_seconds": round(float(latency_seconds or 0.0), 6),
        "cached": bool(cached),
        "status": str(status or "success"),
    }
    _safe_append_jsonl(row)


def record_wall(
    stage: str,
    model: Optional[str] = None,
    wall_seconds: float = 0.0,
    cached: bool = False,
    status: str = "success",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one wall-clock profiling row, input: stage stats -> output: none."""
    if not _enabled:
        return
    ctx = dict(context or _context)
    row = {
        "run_id": ctx.get("run_id") or _run_id,
        "scenario": ctx.get("scenario"),
        "task_id": ctx.get("task_id"),
        "turn": ctx.get("turn"),
        "stage": str(stage),
        "model": model,
        "wall_seconds": round(float(wall_seconds or 0.0), 6),
        "cached": bool(cached),
        "status": str(status or "success"),
    }
    _safe_append_wall_event(row)


def record_task_summary(result: Dict[str, Any]) -> None:
    """Record official task-level totals, input: result dict -> output: none."""
    if not _enabled or not _task_summaries_path or not isinstance(result, dict):
        return
    row = {
        "run_id": _run_id,
        "scenario": _context.get("scenario"),
        "task_id": result.get("task_id", _context.get("task_id")),
        "rounds_count": _to_int(result.get("rounds_count", 0)),
        "input_tokens": _to_int(result.get("input_tokens", 0)),
        "output_tokens": _to_int(result.get("output_tokens", 0)),
        "tool_calls_count": _to_int(result.get("tool_calls_count", 0)),
        "user_response_time_seconds": round(float(result.get("user_response_time_seconds", 0.0) or 0.0), 6),
        "agent_response_time_seconds": round(float(result.get("agent_response_time_seconds", 0.0) or 0.0), 6),
        "execution_time_seconds": round(float(result.get("execution_time_seconds", 0.0) or 0.0), 6),
    }
    _safe_append_task_summary(row)


def summarize() -> Optional[Dict[str, Any]]:
    """Summarize calls.jsonl to CSV/JSON, input: none -> output: summary dict or none."""
    if not _enabled:
        return None
    if (not _calls_path or not _calls_path.exists()) and (not _wall_events_path or not _wall_events_path.exists()):
        return None

    rows = _load_jsonl_rows(_calls_path, "profiling calls")
    wall_rows = _load_jsonl_rows(_wall_events_path, "wall-clock profiling events")
    if not wall_rows and rows:
        wall_rows = _derive_wall_rows_from_call_rows(rows)

    task_summary_rows = _load_task_summary_rows()
    summary_rows = _build_latency_summary_rows(wall_rows, task_summary_rows)
    summary = {
        "run_id": _run_id,
        "total_basis": "agent_response_time_seconds",
        "calls_path": str(_calls_path),
        "wall_events_path": str(_wall_events_path) if _wall_events_path else None,
        "task_summaries_path": str(_task_summaries_path) if _task_summaries_path else None,
        "total_calls": len(rows),
        "total_wall_events": len(wall_rows),
        "latency_stages": summary_rows,
    }
    _write_summary(summary, summary_rows)
    return summary


def _load_task_summary_rows() -> list:
    """Load official task summary rows, input: none -> output: list."""
    if not _task_summaries_path or not _task_summaries_path.exists():
        return []
    rows = []
    try:
        with _task_summaries_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except OSError as exc:
        _warn(f"[stage_latency] Failed to read task summaries: {exc}")
    return rows


def _build_latency_summary_rows(wall_rows: list, task_summary_rows: list) -> list:
    """Build one latency-only table, input: wall/task rows -> output: rows."""
    wall_totals: Dict[str, Dict[str, Any]] = {}

    for row in wall_rows:
        stage = _canonical_stage(row.get("stage"))
        if stage in NON_OFFICIAL_USER_STAGES or stage == UNATTRIBUTED_OVERHEAD_STAGE:
            continue
        bucket = wall_totals.setdefault(stage, {
            "call_count": 0,
            "wall_seconds": 0.0,
        })
        bucket["call_count"] += 1
        bucket["wall_seconds"] += float(row.get("wall_seconds", 0.0) or 0.0)

    extra_stages = sorted(
        stage
        for stage in wall_totals
        if stage not in OFFICIAL_AGENT_STAGES
    )
    stage_order = OFFICIAL_AGENT_STAGES + extra_stages

    official_values = []
    for row in task_summary_rows:
        if "agent_response_time_seconds" not in row:
            continue
        try:
            official_values.append(float(row.get("agent_response_time_seconds", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    official_wall_total = round(sum(official_values), 6)

    rows = []
    visible_wall_total = 0.0
    for stage in stage_order:
        wall_bucket = wall_totals.get(stage, {})
        wall_seconds = round(float(wall_bucket.get("wall_seconds", 0.0) or 0.0), 6)
        visible_wall_total += wall_seconds
        rows.append({
            "stage": stage,
            "call_count": int(wall_bucket.get("call_count", 0) or 0),
            "total_wall_seconds": wall_seconds,
            "percent_of_total_time": 0.0,
        })

    if not official_values:
        official_wall_total = round(visible_wall_total, 6)

    overhead_wall = round(official_wall_total - round(visible_wall_total, 6), 6)
    rows.append({
        "stage": UNATTRIBUTED_OVERHEAD_STAGE,
        "call_count": 1 if abs(overhead_wall) > 0.0000005 else 0,
        "total_wall_seconds": overhead_wall,
        "percent_of_total_time": 0.0,
    })
    for row in rows:
        row["percent_of_total_time"] = _percent(row["total_wall_seconds"], official_wall_total)
    rows.append({
        "stage": "TOTAL",
        "call_count": "",
        "total_wall_seconds": official_wall_total,
        "percent_of_total_time": "100.00" if official_wall_total else "0.00",
    })
    return rows


def _canonical_stage(stage: Any) -> str:
    """Normalize stage aliases, input: stage -> output: canonical stage."""
    text = str(stage or "unknown")
    if text == "visual_agent":
        return "visual_recognition"
    return text


def _load_jsonl_rows(path: Optional[Path], label: str) -> list:
    """Load JSONL rows, input: path/label -> output: list."""
    if not path or not path.exists():
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except OSError as exc:
        _warn(f"[stage_latency] Failed to read {label}: {exc}")
    return rows


def _derive_wall_rows_from_call_rows(rows: list) -> list:
    """Build backward-compatible wall rows from call latency rows."""
    wall_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        wall_rows.append({
            "run_id": row.get("run_id"),
            "scenario": row.get("scenario"),
            "task_id": row.get("task_id"),
            "turn": row.get("turn"),
            "stage": row.get("stage"),
            "model": row.get("model"),
            "wall_seconds": float(row.get("latency_seconds", 0.0) or 0.0),
            "cached": bool(row.get("cached")),
            "status": row.get("status", "success"),
        })
    return wall_rows


def _write_summary(summary: Dict[str, Any], summary_rows: Any) -> None:
    if not _profile_dir:
        return
    try:
        _safe_mkdir(_profile_dir)
        json_path = _profile_dir / "stage_latency_summary.json"
        csv_path = _profile_dir / "stage_latency_summary.csv"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage",
                    "call_count",
                    "total_wall_seconds",
                    "percent_of_total_time",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
    except OSError as exc:
        _warn(f"[stage_latency] Failed to write summary: {exc}")


def _safe_append_jsonl(row: Dict[str, Any]) -> None:
    if not _calls_path:
        return
    try:
        _safe_mkdir(_calls_path.parent)
        with _calls_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        _warn(f"[stage_latency] Failed to write profiling row: {exc}")


def _safe_append_wall_event(row: Dict[str, Any]) -> None:
    if not _wall_events_path:
        return
    try:
        _safe_mkdir(_wall_events_path.parent)
        with _wall_events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        _warn(f"[stage_latency] Failed to write wall-clock profiling row: {exc}")


def _safe_append_task_summary(row: Dict[str, Any]) -> None:
    if not _task_summaries_path:
        return
    try:
        _safe_mkdir(_task_summaries_path.parent)
        with _task_summaries_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        _warn(f"[stage_latency] Failed to write task summary row: {exc}")


def _safe_mkdir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _warn(f"[stage_latency] Failed to create profiling directory: {exc}")


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _percent(value: Any, total: Any) -> str:
    try:
        total_float = float(total or 0.0)
        if total_float == 0.0:
            return "0.00"
        return f"{float(value or 0.0) / total_float * 100.0:.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _warn(message: str) -> None:
    global _warned_write_failure
    if _warned_write_failure:
        return
    _warned_write_failure = True
    print(message, flush=True)
