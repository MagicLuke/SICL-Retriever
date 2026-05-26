import tempfile
import unittest
from pathlib import Path

import numpy as np

from sicl_retriever.io import write_jsonl
from sicl_retriever.validation import summarize_retrieved_manifest, validate_inputs


class ValidationTests(unittest.TestCase):
    def test_validate_reports_embedding_row_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "a.wav"
            audio.write_bytes(b"placeholder")
            manifest = root / "manifest.jsonl"
            embeddings = root / "embeddings.npy"
            write_jsonl([{"audio": str(audio), "text": "x"}], manifest)
            np.save(embeddings, np.ones((2, 2), dtype=np.float32))
            report = validate_inputs(
                input_meta=str(manifest),
                candidate_meta=str(manifest),
                candidate_text_embeddings_path=str(embeddings),
                test_text_embeddings_path=str(embeddings),
            )
            self.assertFalse(report.ok)
            self.assertTrue(any("candidate text embeddings" in error for error in report.errors))

    def test_validate_reports_missing_required_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "a.wav"
            audio.write_bytes(b"placeholder")
            manifest = root / "manifest.jsonl"
            write_jsonl([{"audio": str(audio)}], manifest)
            report = validate_inputs(input_meta=str(manifest), candidate_meta=str(manifest))
            self.assertFalse(report.ok)
            self.assertTrue(any("missing required column 'text'" in error for error in report.errors))

    def test_summarize_retrieved_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "retrieved.jsonl"
            write_jsonl(
                [
                    {
                        "audio": "q.wav",
                        "in_context_examples": [{"audio": "a.wav"}, {"audio": "b.wav"}],
                        "in_context_example_ids": [0, 1],
                    },
                    {"audio": "z.wav", "in_context_examples": []},
                ],
                manifest,
            )
            summary = summarize_retrieved_manifest(str(manifest))
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["total_examples"], 2)
            self.assertEqual(summary["max_examples"], 2)
            self.assertEqual(summary["rows_with_ids"], 1)


if __name__ == "__main__":
    unittest.main()
