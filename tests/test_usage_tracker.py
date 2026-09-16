import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.harness.usage import attribute_agent_loop_usage
from src.usage_tracker import accumulate_usage, empty_usage, extract_usage, record, summarize


class UsageTrackerTests(unittest.TestCase):
    def test_extracts_cached_and_uncached_tokens_from_openai_usage(self):
        usage = SimpleNamespace(
            prompt_tokens=1_000,
            completion_tokens=80,
            total_tokens=1_080,
            prompt_tokens_details=SimpleNamespace(cached_tokens=750),
        )

        tokens = extract_usage(usage)

        self.assertEqual(tokens["prompt_tokens"], 1_000)
        self.assertEqual(tokens["cached_prompt_tokens"], 750)
        self.assertEqual(tokens["uncached_prompt_tokens"], 250)
        self.assertEqual(tokens["unknown_cache_prompt_tokens"], 0)
        self.assertEqual(tokens["completion_tokens"], 80)
        self.assertEqual(tokens["total_tokens"], 1_080)

    def test_supports_raw_dict_usage(self):
        usage = {
            "prompt_tokens": 400,
            "completion_tokens": 20,
            "total_tokens": 420,
            "prompt_tokens_details": {"cached_tokens": 0},
        }

        tokens = extract_usage(usage)

        self.assertEqual(tokens["cached_prompt_tokens"], 0)
        self.assertEqual(tokens["uncached_prompt_tokens"], 400)
        self.assertEqual(tokens["unknown_cache_prompt_tokens"], 0)

    def test_missing_cache_details_are_unknown_not_uncached(self):
        historical_record = {
            "prompt_tokens": 300,
            "completion_tokens": 10,
            "total_tokens": 310,
        }

        tokens = extract_usage(historical_record)

        self.assertEqual(tokens["cached_prompt_tokens"], 0)
        self.assertEqual(tokens["uncached_prompt_tokens"], 0)
        self.assertEqual(tokens["unknown_cache_prompt_tokens"], 300)

    def test_accumulate_and_summarize_preserve_prompt_token_invariant(self):
        current = {
            "agent": "qa",
            "label": "qa_1",
            "prompt_tokens": 1_000,
            "cached_prompt_tokens": 800,
            "uncached_prompt_tokens": 200,
            "unknown_cache_prompt_tokens": 0,
            "completion_tokens": 50,
            "total_tokens": 1_050,
        }
        historical = {
            "agent": "qa",
            "label": "qa_1",
            "prompt_tokens": 300,
            "completion_tokens": 10,
            "total_tokens": 310,
        }

        running = empty_usage()
        accumulate_usage(running, current)
        accumulate_usage(running, historical)
        summary = summarize([current, historical])

        self.assertEqual(running["prompt_tokens"], 1_300)
        self.assertEqual(running["cached_prompt_tokens"], 800)
        self.assertEqual(running["uncached_prompt_tokens"], 200)
        self.assertEqual(running["unknown_cache_prompt_tokens"], 300)
        self.assertEqual(summary["total"], {**running, "calls": 2})
        self.assertEqual(
            running["prompt_tokens"],
            running["cached_prompt_tokens"]
            + running["uncached_prompt_tokens"]
            + running["unknown_cache_prompt_tokens"],
        )

    def test_agent_loop_attribution_ignores_api_cache_details(self):
        first = {
            "prompt_tokens": 1_000,
            "completion_tokens": 80,
            "total_tokens": 1_080,
            "prompt_tokens_details": {"cached_tokens": 900},
        }
        second = {
            "prompt_tokens": 1_260,
            "completion_tokens": 40,
            "total_tokens": 1_300,
            "prompt_tokens_details": {"cached_tokens": 12},
        }

        first_tokens = attribute_agent_loop_usage(first, None)
        second_tokens = attribute_agent_loop_usage(second, first_tokens["prompt_tokens"])

        self.assertEqual(first_tokens["cached_prompt_tokens"], 0)
        self.assertEqual(first_tokens["uncached_prompt_tokens"], 1_000)
        self.assertEqual(second_tokens["cached_prompt_tokens"], 1_000)
        self.assertEqual(second_tokens["uncached_prompt_tokens"], 260)
        self.assertEqual(second_tokens["unknown_cache_prompt_tokens"], 0)

    def test_record_persists_cache_breakdown(self):
        usage = SimpleNamespace(
            prompt_tokens=900,
            completion_tokens=100,
            total_tokens=1_000,
            prompt_tokens_details=SimpleNamespace(cached_tokens=640),
        )

        with tempfile.TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "token_usage.jsonl"
            with patch("src.usage_tracker._usage_path", return_value=usage_path):
                entry = record("novel", "qa", "qa_1", "model", usage)

            saved = json.loads(usage_path.read_text(encoding="utf-8"))

        self.assertEqual(entry["cached_prompt_tokens"], 640)
        self.assertEqual(entry["uncached_prompt_tokens"], 260)
        self.assertEqual(saved["cached_prompt_tokens"], 640)
        self.assertEqual(saved["uncached_prompt_tokens"], 260)


if __name__ == "__main__":
    unittest.main()
