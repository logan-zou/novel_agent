import json
import tempfile
import unittest

from src.novel.builder import _builder_tools
from src.novel.corpus import CorpusStore
from src.novel.knowledge import BuildSession, KnowledgeStore


class BuilderToolTests(unittest.TestCase):
    def test_extraction_exposes_batch_entity_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = CorpusStore(tmp)
            corpus.save_chapters(
                [{"chapter_no": 1, "title": "开始", "text": "张三得到筑基传承。"}]
            )
            session = BuildSession(KnowledgeStore(tmp))
            tools = {tool.name: tool for tool in _builder_tools(session, corpus)}

            self.assertIn("read_entities", tools)
            self.assertIn("upsert_entities", tools)
            self.assertNotIn("read_entity", tools)
            self.assertNotIn("upsert_entity", tools)

            written = tools["upsert_entities"].handler(
                entities=[
                    {
                        "kind": "setting",
                        "name": "筑基",
                        "aliases": [],
                        "summary": "修行境界",
                        "content": "修行的第二境。",
                        "source_chapters": [1],
                        "domain": "力量体系",
                    },
                    {
                        "kind": "character",
                        "name": "张三",
                        "aliases": [],
                        "summary": "获得传承的修士",
                        "content": "得到[[settings/力量体系/筑基]]传承。",
                        "source_chapters": [1],
                    },
                ]
            )
            written_payload = json.loads(written)
            self.assertEqual(written_payload["staged_count"], 2)
            self.assertTrue(tools["upsert_entities"].progress_detector(written))

            read = tools["read_entities"].handler(
                requests=[
                    {"ref": "settings/力量体系/筑基"},
                    {"ref": "characters/张三"},
                ]
            )
            read_payload = json.loads(read)
            self.assertEqual(len(read_payload["entities"]), 2)
            self.assertIn("修行的第二境", read_payload["entities"][0]["content"])


if __name__ == "__main__":
    unittest.main()
