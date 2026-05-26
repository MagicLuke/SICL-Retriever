import tempfile
import unittest
from pathlib import Path

from sicl_retriever.cli import main
from sicl_retriever.io import load_manifest, read_jsonl


class CvEnExampleTests(unittest.TestCase):
    @property
    def example_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "examples" / "cv_en"

    def test_cv_en_audio_files_are_real_and_manifest_relative(self):
        rows = load_manifest(self.example_dir / "test.jsonl")
        rows.extend(load_manifest(self.example_dir / "candidates.jsonl"))
        for row in rows:
            audio = Path(str(row["audio"]))
            self.assertTrue(audio.exists(), audio)
            self.assertEqual(audio.suffix, ".mp3")
            self.assertGreater(audio.stat().st_size, 1000)

        raw_rows = read_jsonl(self.example_dir / "test.jsonl") + read_jsonl(self.example_dir / "candidates.jsonl")
        for row in raw_rows:
            self.assertFalse(Path(str(row["audio"])).is_absolute())

    def test_cv_en_validate_and_retrieve_are_deterministic(self):
        rc = main(
            [
                "validate",
                "--input-meta",
                str(self.example_dir / "test.jsonl"),
                "--candidate-meta",
                str(self.example_dir / "candidates.jsonl"),
                "--candidate-text-embeddings",
                str(self.example_dir / "candidate_text.npy"),
                "--test-text-embeddings",
                str(self.example_dir / "test_text.npy"),
            ]
        )
        self.assertEqual(rc, 0)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cv_en_ticl.jsonl"
            rc = main(
                [
                    "retrieve",
                    "--method",
                    "ticl",
                    "--input-meta",
                    str(self.example_dir / "test.jsonl"),
                    "--candidate-meta",
                    str(self.example_dir / "candidates.jsonl"),
                    "--output-meta",
                    str(output),
                    "--candidate-text-embeddings",
                    str(self.example_dir / "candidate_text.npy"),
                    "--test-text-embeddings",
                    str(self.example_dir / "test_text.npy"),
                    "--topk",
                    "3",
                    "--include-ids",
                ]
            )
            self.assertEqual(rc, 0)
            row = read_jsonl(output)[0]
            expected = read_jsonl(self.example_dir / "expected_ticl.jsonl")[0]
            self.assertEqual(sorted(row), ["audio", "in_context_example_ids", "in_context_examples", "text"])
            self.assertEqual(row["text"], expected["text"])
            self.assertEqual(Path(str(row["audio"])).name, Path(str(expected["audio"])).name)
            self.assertEqual(row["in_context_example_ids"], expected["in_context_example_ids"])
            for actual_item, expected_item in zip(row["in_context_examples"], expected["in_context_examples"]):
                self.assertEqual(sorted(actual_item), ["audio", "text"])
                self.assertEqual(actual_item["text"], expected_item["text"])
                self.assertEqual(Path(str(actual_item["audio"])).name, Path(str(expected_item["audio"])).name)


if __name__ == "__main__":
    unittest.main()
