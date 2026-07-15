"""Regression tests for official per-stage latency profiling."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from run import boxed_reviewer, stage_latency


class StageLatencyTests(unittest.TestCase):
    def tearDown(self):
        stage_latency.configure(enabled=False)

    def _configure(self, root: Path, run_id: str = "latency-test") -> Path:
        stage_latency.configure(enabled=True, run_id=run_id, profile_root=str(root))
        stage_latency.set_context(scenario="retail1", task_id=1, turn=0)
        return root / run_id

    def _boxed_frame(self, root: Path):
        image_path = root / "boxed.jpg"
        Image.new("RGB", (120, 80), color=(30, 40, 50)).save(image_path)
        return {
            "path": str(image_path),
            "second": 2,
            "boxes": [{"box": [10, 10, 70, 70], "desc": "candidate"}],
        }

    def test_custom_profile_name_preserves_run_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "gpt-5.5" / "image_base64" / "retail1"
            profile_name = "20260710_175435_retail1_easy"
            stage_latency.configure(
                enabled=True,
                run_id="20260710_175435",
                profile_root=str(root),
                profile_name=profile_name,
            )
            stage_latency.set_context(scenario="retail1", task_id=1, turn=0)
            stage_latency.record_wall("supervisor", wall_seconds=1.0)
            summary = stage_latency.summarize()

            profile_dir = root / profile_name
            wall_event = json.loads(
                (profile_dir / "wall_events.jsonl").read_text(encoding="utf-8").strip()
            )

        self.assertEqual(wall_event["run_id"], "20260710_175435")
        self.assertEqual(summary["run_id"], "20260710_175435")
        self.assertEqual(Path(summary["wall_events_path"]), profile_dir / "wall_events.jsonl")

    def test_nested_reviewer_is_exclusive_and_total_uses_official_agent_time(self):
        with tempfile.TemporaryDirectory() as temp:
            profile_dir = self._configure(Path(temp))
            stage_latency.record_wall("user_agent", wall_seconds=20.0)

            with patch.object(stage_latency.time, "perf_counter", side_effect=[0.0, 2.0, 5.0, 10.0]):
                boxer = stage_latency.start("frame_boxer", model="visual_boxed_frames")
                reviewer = stage_latency.start("frames_reviewer", model="review-model")
                stage_latency.end(reviewer, input_tokens=12, output_tokens=4)
                stage_latency.end(boxer)

            stage_latency.record_task_summary({
                "task_id": 1,
                "user_response_time_seconds": 20.0,
                "agent_response_time_seconds": 10.0,
                "execution_time_seconds": 30.0,
            })
            summary = stage_latency.summarize()

            calls = [
                json.loads(line)
                for line in (profile_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        calls_by_stage = {row["stage"]: row for row in calls}
        self.assertEqual(calls_by_stage["frames_reviewer"]["latency_seconds"], 3.0)
        self.assertEqual(calls_by_stage["frames_reviewer"]["input_tokens"], 12)
        self.assertEqual(calls_by_stage["frames_reviewer"]["output_tokens"], 4)
        self.assertEqual(calls_by_stage["frame_boxer"]["latency_seconds"], 7.0)

        rows_by_stage = {row["stage"]: row for row in summary["latency_stages"]}
        self.assertEqual(summary["total_basis"], "agent_response_time_seconds")
        self.assertNotIn("user_agent", rows_by_stage)
        self.assertEqual(rows_by_stage["frame_boxer"]["total_wall_seconds"], 7.0)
        self.assertEqual(rows_by_stage["frames_reviewer"]["total_wall_seconds"], 3.0)
        self.assertEqual(rows_by_stage["unattributed_overhead"]["total_wall_seconds"], 0.0)
        self.assertEqual(rows_by_stage["TOTAL"]["total_wall_seconds"], 10.0)

    def test_multiple_reviewer_calls_are_each_subtracted_from_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            self._configure(Path(temp))
            with patch.object(
                stage_latency.time,
                "perf_counter",
                side_effect=[0.0, 1.0, 3.0, 4.0, 7.0, 10.0],
            ):
                boxer = stage_latency.start("frame_boxer")
                first = stage_latency.start("frames_reviewer")
                stage_latency.end(first)
                second = stage_latency.start("frames_reviewer")
                stage_latency.end(second)
                stage_latency.end(boxer)
            summary = stage_latency.summarize()

        rows_by_stage = {row["stage"]: row for row in summary["latency_stages"]}
        self.assertEqual(rows_by_stage["frame_boxer"]["total_wall_seconds"], 5.0)
        self.assertEqual(rows_by_stage["frames_reviewer"]["call_count"], 2)
        self.assertEqual(rows_by_stage["frames_reviewer"]["total_wall_seconds"], 5.0)
        self.assertEqual(rows_by_stage["TOTAL"]["total_wall_seconds"], 10.0)

    def test_reviewer_records_tokens_and_closes_error_handle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile_dir = self._configure(root)
            boxed_frame = self._boxed_frame(root)
            response = '{"verdict":"accept","corrected_bbox":null,"reason":"ok"}'
            with patch.object(
                boxed_reviewer,
                "call_frames_review_model",
                return_value=(response, 21, 6),
            ):
                result = boxed_reviewer.review_boxed_frame(boxed_frame, "the bottle")

            with patch.object(
                boxed_reviewer,
                "call_frames_review_model",
                side_effect=RuntimeError("review failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "review failed"):
                    boxed_reviewer.review_boxed_frame(boxed_frame, "the bottle")

            with patch.object(
                boxed_reviewer,
                "call_frames_review_model",
                return_value=("Error: reviewer unavailable", 0, 0),
            ):
                failed_result = boxed_reviewer.review_boxed_frame(boxed_frame, "the bottle")

            calls = [
                json.loads(line)
                for line in (profile_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["verdict"], "accept")
        self.assertEqual(calls[0]["stage"], "frames_reviewer")
        self.assertEqual(calls[0]["model"], boxed_reviewer.FRAMES_REVIEW_MODEL_NAME)
        self.assertEqual(calls[0]["input_tokens"], 21)
        self.assertEqual(calls[0]["output_tokens"], 6)
        self.assertEqual(calls[0]["status"], "success")
        self.assertEqual(calls[1]["status"], "error")
        self.assertEqual(failed_result["verdict"], "reject")
        self.assertEqual(calls[2]["status"], "error")
        self.assertEqual(stage_latency._active_handles, [])


if __name__ == "__main__":
    unittest.main()
