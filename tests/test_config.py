import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import ConfigurationError, _load_env_file, require_llm_model


class ConfigurationTests(unittest.TestCase):
    def test_env_file_loads_values_without_overriding_process_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "# comment\nNOVEL_AGENT_TEST_NEW='from file'\n"
                "NOVEL_AGENT_TEST_EXISTING=from-file\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"NOVEL_AGENT_TEST_EXISTING": "from-process"},
                clear=False,
            ):
                os.environ.pop("NOVEL_AGENT_TEST_NEW", None)
                _load_env_file(env_path)
                self.assertEqual(os.environ["NOVEL_AGENT_TEST_NEW"], "from file")
                self.assertEqual(os.environ["NOVEL_AGENT_TEST_EXISTING"], "from-process")
                os.environ.pop("NOVEL_AGENT_TEST_NEW", None)

    def test_model_configuration_has_actionable_error(self):
        with self.assertRaisesRegex(ConfigurationError, "LLM_MODEL"):
            require_llm_model("", "QA_MODEL")
        self.assertEqual(require_llm_model("model-name", "QA_MODEL"), "model-name")


if __name__ == "__main__":
    unittest.main()
