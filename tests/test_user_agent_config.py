import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
READ_MODEL_NAME = """
import sys
import types

openai = types.ModuleType("openai")
openai.OpenAI = object
sys.modules["openai"] = openai

from config.user_agent_config import USER_MODEL_NAME

print(USER_MODEL_NAME)
"""


class UserModelNameConfigTest(unittest.TestCase):
    def read_model_name(self, override=None):
        env = os.environ.copy()
        env.pop("USER_MODEL_NAME", None)
        if override is not None:
            env["USER_MODEL_NAME"] = override

        result = subprocess.run(
            [sys.executable, "-c", READ_MODEL_NAME],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_uses_existing_default_when_environment_is_unset(self):
        self.assertEqual(
            self.read_model_name(),
            "Qwen/Qwen3.5-397B-A17B",
        )

    def test_uses_user_model_name_from_environment(self):
        self.assertEqual(
            self.read_model_name("example/custom-model"),
            "example/custom-model",
        )


if __name__ == "__main__":
    unittest.main()
