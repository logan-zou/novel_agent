import json
import tempfile
import unittest

from src.novel.corpus import CorpusStore


class CorpusStoreTests(unittest.TestCase):
    def test_search_supports_latest_first_all_mode_and_pagination(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CorpusStore(tmp)
            store.save_chapters(
                [
                    {"chapter_no": index, "title": str(index), "text": f"角色 S级 第{index}次揭示"}
                    for index in range(1, 31)
                ]
            )
            first = json.loads(store.search(["角色", "S级"], mode="all", order="desc", limit=3))
            second = json.loads(
                store.search(
                    ["角色", "S级"],
                    mode="all",
                    order="desc",
                    limit=3,
                    cursor=first["next_cursor"],
                )
            )

        self.assertEqual([item["chapter"] for item in first["matches"]], [30, 29, 28])
        self.assertEqual([item["chapter"] for item in second["matches"]], [27, 26, 25])
        self.assertTrue(first["has_more"])


if __name__ == "__main__":
    unittest.main()
