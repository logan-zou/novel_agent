import unittest

from src.preprocess import make_batches


def _chapters(count: int, chars: int = 40) -> list[dict]:
    return [
        {"chapter_no": index + 1, "title": str(index + 1), "text": "x" * chars}
        for index in range(count)
    ]


class BatchTests(unittest.TestCase):
    def test_omitted_minimum_keeps_fixed_window_behavior(self):
        batches = make_batches(_chapters(6), 100)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 2])

    def test_progressive_window_reduces_later_batch_size(self):
        batches = make_batches(_chapters(10), 100, 50)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 2, 1, 1, 1, 1])
        self.assertEqual(
            [chapter["chapter_no"] for batch in batches for chapter in batch],
            list(range(1, 11)),
        )

    def test_tail_replanning_accounts_for_completed_text(self):
        chapters = _chapters(10)
        batches = make_batches(
            chapters[4:],
            100,
            50,
            processed_chars=160,
            total_chars=400,
        )
        self.assertEqual(batches[0][0]["chapter_no"], 5)
        self.assertEqual([len(batch) for batch in batches], [2, 1, 1, 1, 1])

    def test_rejects_invalid_window_range(self):
        with self.assertRaises(ValueError):
            make_batches(_chapters(1), 100, 101)


if __name__ == "__main__":
    unittest.main()
