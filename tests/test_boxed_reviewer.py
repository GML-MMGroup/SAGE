"""Regression tests for the Gemini boxed-frame reviewer."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from config import frames_review_config
from run import boxed_reviewer, prompts, visual_boxed_frames


class _RecordingCompletions:
    def __init__(self, content):
        self.content = content
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=None,
        )


class BoxedReviewerTests(unittest.TestCase):
    def _source_image(self, root: Path, name: str = "source.jpg") -> Path:
        path = root / name
        Image.new("RGB", (120, 80), color=(30, 40, 50)).save(path)
        return path

    def _candidate(self, path: Path, idx: int = 1, second: int = 2, box=None):
        return {
            "idx": idx,
            "path": str(path),
            "second": second,
            "size": (120, 80),
            "box": list(box or [10.0, 10.0, 70.0, 70.0]),
            "desc": "visible candidate",
            "evidence": "appearance",
            "locator_certainty": "confident",
            "model": "mock-locator",
        }

    def test_config_uses_vapi_gpt_default_reasoning_and_seed(self):
        completions = _RecordingCompletions('{"verdict":"accept","corrected_bbox":null,"reason":"ok"}')
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with (
            patch.object(frames_review_config, "FRAMES_REVIEW_API_KEY", "vapi-key"),
            patch.object(frames_review_config, "FRAMES_REVIEW_API_BASE_URL", "https://vapi.example/v1"),
            patch.object(frames_review_config, "_build_openai_client", return_value=client) as build_client,
        ):
            response, _, _ = frames_review_config.call_frames_review_model([], max_retries=1)

        self.assertIn('"accept"', response)
        build_client.assert_called_once_with("vapi-key", "https://vapi.example/v1")
        request = completions.requests[0]
        self.assertEqual(request["model"], "gpt-5.6-sol")
        self.assertEqual(request["seed"], 66)
        self.assertEqual(request["temperature"], 0)
        self.assertIsNone(frames_review_config.FRAMES_REVIEW_REASONING_EFFORT)
        self.assertNotIn("extra_body", request)
        self.assertNotIn("reasoning_effort", request)
        self.assertNotIn("response_format", request)

    def test_system_prompt_is_in_config_and_rules_are_in_run_prompts(self):
        self.assertIn("boxed-frame review and correction agent", frames_review_config.FRAMES_REVIEW_SYSTEM_PROMPT)
        self.assertIn("## Semantic Review Rule", prompts.VISUAL_BOXED_REVIEW_PROMPT)
        self.assertEqual(prompts.VISUAL_BOXED_REVIEW_PROMPT.count("## Pointing Geometry Rule"), 1)
        self.assertIn("trace the visible index finger", prompts.VISUAL_BOXED_REVIEW_PROMPT)
        self.assertIn("return `reject` and do not guess", prompts.VISUAL_BOXED_REVIEW_PROMPT)
        self.assertIn("return `correct` only when another pointed-at target", prompts.VISUAL_BOXED_REVIEW_PROMPT)
        self.assertIn("## Boundary Review Rule", prompts.VISUAL_BOXED_REVIEW_PROMPT)
        self.assertIn('"verdict":"correct"', prompts.VISUAL_BOXED_REVIEW_PROMPT)

    def test_reviewer_receives_one_full_boxed_image_without_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_image(root)
            boxed_path = root / "boxed_001_01_frame_0002s.jpg"
            box_item = {"box": [10.0, 10.0, 70.0, 70.0], "label": "Target", "desc": "candidate"}
            visual_boxed_frames._draw_boxes(str(source), [box_item], boxed_path)
            boxed_frame = {
                "path": str(boxed_path),
                "source_path": str(source),
                "second": 2,
                "boxes": [box_item],
            }
            calls = []

            def fake_call(messages, max_retries=None):
                calls.append(messages)
                return '{"verdict":"accept","corrected_bbox":null,"reason":"tight"}', 0, 0

            with patch.object(boxed_reviewer, "call_frames_review_model", side_effect=fake_call):
                result = boxed_reviewer.review_boxed_frame(boxed_frame, "the pointed bottle")

        self.assertEqual(result["verdict"], "accept")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["content"], boxed_reviewer.FRAMES_REVIEW_SYSTEM_PROMPT)
        user_content = calls[0][1]["content"]
        image_items = [item for item in user_content if item.get("type") == "image_url"]
        self.assertEqual(len(image_items), 1)
        self.assertIn("boxed frame", user_content[0]["text"].lower())
        self.assertNotIn("crop image", user_content[0]["text"].lower())

    def test_invalid_reviewer_response_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_image(root)
            boxed_path = root / "boxed_001_01_frame_0002s.jpg"
            box_item = {"box": [10.0, 10.0, 70.0, 70.0], "label": "Target", "desc": "candidate"}
            visual_boxed_frames._draw_boxes(str(source), [box_item], boxed_path)
            boxed_frame = {"path": str(boxed_path), "second": 2, "boxes": [box_item]}
            with patch.object(boxed_reviewer, "call_frames_review_model", return_value=("not json", 0, 0)):
                result = boxed_reviewer.review_boxed_frame(boxed_frame, "the bottle")

        self.assertEqual(result["verdict"], "reject")
        self.assertEqual(result["corrected_bbox"], None)
        self.assertEqual(result["reason"], "invalid_or_missing_reviewer_verdict")

    def test_corrected_bbox_overwrites_same_boxed_file_then_generates_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_image(root)
            review = {
                "verdict": "correct",
                "corrected_bbox": [40, 5, 90, 75],
                "reason": "shifted to the single correct object",
                "model": "gemini-2.5-pro",
                "raw_response": "{}",
            }
            with (
                patch.object(visual_boxed_frames, "BOX_ROOT", root / "boxed_root"),
                patch.object(visual_boxed_frames, "_locate_selected", return_value=[self._candidate(source)]),
                patch.object(visual_boxed_frames, "review_boxed_frame", return_value=review) as reviewer,
            ):
                sent, manifest = visual_boxed_frames.annotate_visual_image_base64_frames(
                    selected_infos=[{"path": str(source), "second": 2}],
                    all_frame_infos=None,
                    target_text="the correct bottle",
                    final_scenario="retail1",
                    final_task_id=3,
                    final_request_seq=1,
                    target_cardinality={"cardinality": "single", "max_targets": 1},
                )
                request_dir = root / "boxed_root" / "retail1" / "task3" / "request_001"
                names = {path.name for path in request_dir.iterdir()}
                boxed_path = Path(manifest["boxed_main_frames"][0]["path"])
                with Image.open(boxed_path) as image:
                    old_border_pixel = image.getpixel((10, 10))

        reviewer.assert_called_once()
        self.assertEqual(manifest["status"], "boxed")
        self.assertEqual(manifest["review_status"], "corrected")
        self.assertEqual(manifest["box_reviews"][0]["original_bbox"], [10.0, 10.0, 70.0, 70.0])
        self.assertEqual(manifest["box_reviews"][0]["final_bbox"], [40.0, 5.0, 90.0, 75.0])
        self.assertEqual(manifest["boxed_main_frames"][0]["boxes"][0]["box"], [40.0, 5.0, 90.0, 75.0])
        self.assertEqual([item["kind"] for item in sent], ["target_crop", "boxed_frame"])
        self.assertEqual(
            names,
            {"boxed_001_01_frame_0002s.jpg", "crop_001_01_frame_0002s.jpg", "manifest.json"},
        )
        self.assertFalse(old_border_pixel[1] > old_border_pixel[0] + 70)

    def test_selector_auxiliary_frames_bypass_locator_and_reviewer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_image(root, "anchor.jpg")
            auxiliary = self._source_image(root, "auxiliary.jpg")
            review = {
                "verdict": "accept",
                "corrected_bbox": None,
                "reason": "anchor is correct",
                "model": "gpt-5.6-sol",
                "raw_response": "{}",
            }
            with (
                patch.object(visual_boxed_frames, "BOX_ROOT", root / "boxed_root"),
                patch.object(visual_boxed_frames, "_locate_selected", return_value=[self._candidate(source)]),
                patch.object(visual_boxed_frames, "review_boxed_frame", return_value=review) as reviewer,
            ):
                sent, manifest = visual_boxed_frames.annotate_visual_image_base64_frames(
                    selected_infos=[{"path": str(source), "second": 2}],
                    auxiliary_infos=[{"path": str(auxiliary), "second": 3}],
                    all_frame_infos=None,
                    target_text="the pointed bottle",
                    final_scenario="retail1",
                    final_task_id=3,
                    final_request_seq=1,
                    target_cardinality={"cardinality": "single", "max_targets": 1},
                )

        reviewer.assert_called_once()
        self.assertEqual(manifest["review_status"], "accepted")
        self.assertEqual([item["kind"] for item in sent], ["target_crop", "boxed_frame", "auxiliary_frame"])
        self.assertEqual(len(manifest["auxiliary_frames"]), 1)
        self.assertTrue(Path(manifest["auxiliary_frames"][0]["path"]).name.startswith("auxiliary_"))

    def test_any_reject_removes_all_boxed_files_and_falls_back_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_one = self._source_image(root, "one.jpg")
            source_two = self._source_image(root, "two.jpg")
            candidates = [
                self._candidate(source_one, idx=1, second=1),
                self._candidate(source_two, idx=2, second=2),
            ]
            reviews = [
                {"verdict": "accept", "corrected_bbox": None, "reason": "ok", "model": "gemini-2.5-pro"},
                {"verdict": "reject", "corrected_bbox": None, "reason": "ambiguous", "model": "gemini-2.5-pro"},
            ]
            with (
                patch.object(visual_boxed_frames, "BOX_ROOT", root / "boxed_root"),
                patch.object(visual_boxed_frames, "_locate_selected", return_value=candidates),
                patch.object(visual_boxed_frames, "review_boxed_frame", side_effect=reviews) as reviewer,
            ):
                sent, manifest = visual_boxed_frames.annotate_visual_image_base64_frames(
                    selected_infos=[
                        {"path": str(source_one), "second": 1},
                        {"path": str(source_two), "second": 2},
                    ],
                    all_frame_infos=None,
                    target_text="the two bottles",
                    final_scenario="retail1",
                    final_task_id=3,
                    final_request_seq=1,
                    target_cardinality={"cardinality": "multiple", "max_targets": 2},
                )
                request_dir = root / "boxed_root" / "retail1" / "task3" / "request_001"
                names = {path.name for path in request_dir.iterdir()}

        self.assertEqual(reviewer.call_count, 2)
        self.assertEqual(manifest["status"], "box_review_rejected_fallback_original")
        self.assertEqual(manifest["review_status"], "rejected")
        self.assertEqual(manifest["boxed_main_frames"], [])
        self.assertEqual([item["kind"] for item in sent], ["selected_frame", "selected_frame"])
        self.assertEqual(
            names,
            {
                "selected_original_001_frame_0001s.jpg",
                "selected_original_002_frame_0002s.jpg",
                "manifest.json",
            },
        )

    def test_invalid_corrected_bbox_rejects_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_image(root)
            review = {
                "verdict": "correct",
                "corrected_bbox": [0, 0, 1, 1],
                "reason": "bad coordinates",
                "model": "gemini-2.5-pro",
            }
            with (
                patch.object(visual_boxed_frames, "BOX_ROOT", root / "boxed_root"),
                patch.object(visual_boxed_frames, "_locate_selected", return_value=[self._candidate(source)]),
                patch.object(visual_boxed_frames, "review_boxed_frame", return_value=review),
            ):
                sent, manifest = visual_boxed_frames.annotate_visual_image_base64_frames(
                    selected_infos=[{"path": str(source), "second": 2}],
                    all_frame_infos=None,
                    target_text="the bottle",
                    final_scenario="retail1",
                    final_task_id=3,
                    final_request_seq=1,
                    target_cardinality={"cardinality": "single", "max_targets": 1},
                )

        self.assertEqual(manifest["review_status"], "rejected")
        self.assertEqual(manifest["box_reviews"][0]["verdict"], "reject")
        self.assertEqual([item["kind"] for item in sent], ["selected_frame"])

    def test_uncertain_locator_candidate_does_not_call_reviewer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_image(root)
            candidate = self._candidate(source)
            candidate["locator_certainty"] = "uncertain"
            with (
                patch.object(visual_boxed_frames, "BOX_ROOT", root / "boxed_root"),
                patch.object(visual_boxed_frames, "_locate_selected", return_value=[candidate]),
                patch.object(visual_boxed_frames, "review_boxed_frame") as reviewer,
            ):
                sent, manifest = visual_boxed_frames.annotate_visual_image_base64_frames(
                    selected_infos=[{"path": str(source), "second": 2}],
                    all_frame_infos=None,
                    target_text="the bottle",
                    final_scenario="retail1",
                    final_task_id=3,
                    final_request_seq=1,
                    target_cardinality={"cardinality": "single", "max_targets": 1},
                )

        reviewer.assert_not_called()
        self.assertEqual(manifest["review_status"], "not_run")
        self.assertEqual(manifest["status"], "locator_uncertain_fallback_original")
        self.assertEqual(manifest["selected_candidates"], [])
        self.assertEqual([item["kind"] for item in sent], ["selected_frame"])

    def test_mixed_confident_and_uncertain_candidates_fall_back_entire_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_one = self._source_image(root, "one.jpg")
            source_two = self._source_image(root, "two.jpg")
            confident = self._candidate(source_one, idx=1, second=1)
            uncertain = self._candidate(source_two, idx=2, second=2)
            uncertain["locator_certainty"] = "uncertain"
            with (
                patch.object(visual_boxed_frames, "BOX_ROOT", root / "boxed_root"),
                patch.object(visual_boxed_frames, "_locate_selected", return_value=[confident, uncertain]),
                patch.object(visual_boxed_frames, "review_boxed_frame") as reviewer,
            ):
                sent, manifest = visual_boxed_frames.annotate_visual_image_base64_frames(
                    selected_infos=[
                        {"path": str(source_one), "second": 1},
                        {"path": str(source_two), "second": 2},
                    ],
                    all_frame_infos=None,
                    target_text="the two bottles",
                    final_scenario="retail1",
                    final_task_id=3,
                    final_request_seq=1,
                    target_cardinality={"cardinality": "multiple", "max_targets": 2},
                )
                request_dir = root / "boxed_root" / "retail1" / "task3" / "request_001"
                names = {path.name for path in request_dir.iterdir()}

        reviewer.assert_not_called()
        self.assertEqual(manifest["status"], "locator_uncertain_fallback_original")
        self.assertEqual(manifest["review_status"], "not_run")
        self.assertEqual(len(manifest["raw_candidates"]), 2)
        self.assertEqual(len(manifest["accepted_candidates"]), 1)
        self.assertEqual(manifest["selected_candidates"], [])
        self.assertEqual(manifest["boxed_main_frames"], [])
        self.assertEqual([item["kind"] for item in sent], ["selected_frame", "selected_frame"])
        self.assertEqual(
            names,
            {
                "selected_original_001_frame_0001s.jpg",
                "selected_original_002_frame_0002s.jpg",
                "manifest.json",
            },
        )

    def test_cache_with_nonconfident_rendered_box_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            save_dir = Path(temp)
            cached_image = self._source_image(save_dir, "boxed.jpg")
            signature = {"version": visual_boxed_frames.BOX_VERSION}
            manifest = {
                "signature": signature,
                "boxed_main_frames": [{
                    "path": str(cached_image),
                    "boxes": [{"box": [10, 10, 70, 70], "locator_certainty": "uncertain"}],
                }],
                "sent_frames": [{
                    "path": str(cached_image),
                    "kind": "boxed_frame",
                    "boxes": [{"box": [10, 10, 70, 70], "locator_certainty": "uncertain"}],
                }],
            }
            (save_dir / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            infos, cached_manifest = visual_boxed_frames._load_cached(save_dir, signature)

        self.assertIsNone(infos)
        self.assertIsNone(cached_manifest)


if __name__ == "__main__":
    unittest.main()
