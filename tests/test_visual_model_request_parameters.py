"""Regression tests for visual GPT reasoning and sampling parameters."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import frames_selecter_config, visual_agent_config, visual_boxed_config


class _RecordingCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )


def _client_with(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class VisualModelRequestParameterTests(unittest.TestCase):
    def _assert_gpt_defaults(self, request):
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(request["seed"], 66)
        self.assertNotIn("temperature", request)

    def test_frame_selector_uses_default_reasoning_and_seed(self):
        completions = _RecordingCompletions()
        with patch.object(
            frames_selecter_config,
            "_build_openai_client",
            return_value=_client_with(completions),
        ):
            frames_selecter_config.call_frame_selecter_model([], max_retries=1)

        self.assertIsNone(frames_selecter_config.FRAME_SELECTER_REASONING_EFFORT)
        self._assert_gpt_defaults(completions.requests[0])

    def test_frame_selector_strips_legacy_visual_task_wrapper_locally(self):
        from run import utils

        wrapped = "Please identify the visual target. Task: the target bottle"
        self.assertEqual(utils.strip_wrapper(wrapped), "the target bottle")
        self.assertEqual(utils.strip_wrapper("the target bottle"), "the target bottle")

        messages = utils._build_frame_selecter_messages(wrapped, [])
        user_prompt = messages[1]["content"][0]["text"]
        self.assertTrue(user_prompt.startswith("Task: the target bottle\n"))
        self.assertNotIn("Please identify the visual target", user_prompt)

    def test_frame_selector_parses_primary_and_auxiliary_frames(self):
        from run import utils

        frames, auxiliary, cardinality = utils._parse_frame_selection(
            '{"frames":[2,4,7],"auxiliary_frames":[3,5,8,2],"cardinality":"multiple","max_targets":3}',
            10,
        )

        self.assertEqual(frames, [2, 4, 7])
        self.assertEqual(auxiliary, [3, 5, 8])
        self.assertEqual(cardinality["cardinality"], "multiple")
        self.assertEqual(cardinality["max_targets"], 3)

    def test_frame_selector_behavior_rules_are_in_system_message_only(self):
        from run import prompts, utils

        messages = utils._build_frame_selecter_messages("the target bottle", [])
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"][0]["text"]
        moved_sections = (
            "## Chronological Action Order Rule",
            "## Visibility Rule",
            "## Primary And Auxiliary Frame Rule",
            "## Preference Rule",
        )

        for section in moved_sections:
            self.assertIn(section, system_prompt)
            self.assertNotIn(section, user_prompt)
            self.assertNotIn(section, prompts.FRAME_SELECTER_PROMPT)

        self.assertIn("## Menu Rule", user_prompt)
        self.assertIn("## Object Localization Rule", user_prompt)
        self.assertIn("## Target Cardinality Rule", user_prompt)
        self.assertIn("## Output Rule", user_prompt)
        self.assertIn('"auxiliary_frames"', user_prompt)

    def test_visual_boxed_uses_default_reasoning_and_seed(self):
        completions = _RecordingCompletions()
        with patch.object(
            visual_boxed_config,
            "_build_openai_client",
            return_value=_client_with(completions),
        ):
            visual_boxed_config.call_visual_boxed_model(
                [], model_name="gpt-5.5", max_retries=1
            )

        self.assertIsNone(visual_boxed_config.VISUAL_BOXED_REASONING_EFFORT)
        self._assert_gpt_defaults(completions.requests[0])

    def test_visual_boxed_grounding_rules_are_in_system_message_only(self):
        from run import prompts, visual_boxed_frames

        requests = []

        def fake_call(messages, **kwargs):
            requests.append(messages)
            return '{"targets":[]}', 0, 0

        with (
            patch.object(visual_boxed_frames, "VISUAL_BOXED_MODEL_NAMES", ["mock-grounder"]),
            patch.object(visual_boxed_frames, "call_visual_boxed_model", side_effect=fake_call),
        ):
            visual_boxed_frames._model_call([
                {"type": "text", "text": prompts.VISUAL_BOXED_LOCATOR_PROMPT}
            ])

        system_prompt = requests[0][0]["content"]
        user_prompt = requests[0][1]["content"][0]["text"]
        moved_rules = (
            "Each bbox must tightly cover the requested target",
            "A bbox must contain exactly one target object",
            "## Label And Text Rule",
            "## Pointing Rule",
            "## Binary Certainty Rule",
        )

        for rule in moved_rules:
            self.assertIn(rule, system_prompt)
            self.assertNotIn(rule, user_prompt)

        self.assertIn("## Frame Context", user_prompt)
        self.assertIn("## Menu And Kitchen Rule", user_prompt)
        self.assertIn("## BBox Rule", user_prompt)
        self.assertIn("## Output Rule", user_prompt)
        self.assertEqual(
            visual_boxed_frames.BOX_VERSION,
            "visual_box_v13_system_grounding_rules",
        )

    def test_visual_agent_uses_default_reasoning_and_seed(self):
        completions = _RecordingCompletions()
        with patch.object(
            visual_agent_config,
            "OpenAI",
            return_value=_client_with(completions),
        ):
            visual_agent_config.call_visual_agent_model([], max_retries=1)

        self.assertIsNone(visual_agent_config.VISUAL_AGENT_REASONING_EFFORT)
        self._assert_gpt_defaults(completions.requests[0])

    def test_visual_boxed_qwen_fallback_keeps_temperature_zero(self):
        completions = _RecordingCompletions()
        with patch.object(
            visual_boxed_config,
            "_build_openai_client",
            return_value=_client_with(completions),
        ):
            visual_boxed_config.call_visual_boxed_model(
                [], model_name="qwen-vl-max", max_retries=1
            )

        request = completions.requests[0]
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["seed"], 66)
        self.assertNotIn("reasoning_effort", request)
        self.assertNotIn("extra_body", request)


if __name__ == "__main__":
    unittest.main()
