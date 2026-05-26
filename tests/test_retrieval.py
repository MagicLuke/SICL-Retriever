import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sicl_retriever.cli import build_parser, main
from sicl_retriever.io import load_manifest, read_jsonl, write_jsonl
from sicl_retriever.retrieval import TICLPlusRetriever, TICLRetriever, attach_ice, retrieve_to_manifest


class RetrievalTests(unittest.TestCase):
    def test_text_retrieval_inner_product(self):
        candidates = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        ids = TICLRetriever(candidates, metric="IP").retrieve_ids(queries, topk=2, batch_size=1)
        self.assertEqual(ids.shape, (1, 2))
        self.assertEqual(ids[0, 0], 0)
        self.assertEqual(ids[0, 1], 2)

    def test_text_retrieval_ties_sort_by_candidate_id(self):
        candidates = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        ids = TICLRetriever(candidates, metric="IP").retrieve_ids(queries, topk=2, batch_size=1)
        self.assertEqual(ids.tolist(), [[0, 1]])

    def test_ticl_plus_reranks_text_candidates_by_audio(self):
        candidate_text = np.array([[1.0, 0.0], [0.99, 0.0], [0.98, 0.0]], dtype=np.float32)
        query_text = np.array([[1.0, 0.0]], dtype=np.float32)
        candidate_audio = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        query_audio = np.array([[0.0, 1.0]], dtype=np.float32)
        ids = TICLPlusRetriever(
            candidate_audio,
            candidate_text,
            metric="IP",
            candidate_multiplier=3,
        ).retrieve_ids(query_text, query_audio, topk=2, batch_size=1)
        self.assertEqual(ids.tolist(), [[1, 2]])

    def test_attach_ice_removes_self_match(self):
        test_rows = [{"audio": "/a.wav", "text": "query"}]
        candidates = [
            {"audio": "/a.wav", "text": "self"},
            {"audio": "/b.wav", "text": "b"},
            {"audio": "/c.wav", "text": "c"},
        ]
        rows = attach_ice(test_rows, candidates, np.array([[0, 1, 2]]), topk=2, ids_column="ice_ids")
        self.assertEqual([ice["text"] for ice in rows[0]["in_context_examples"]], ["b", "c"])
        self.assertEqual(rows[0]["ice_ids"], [1, 2])

    def test_retrieve_to_manifest_validates_embedding_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "a.wav"
            audio.write_bytes(b"placeholder")
            input_meta = tmp_path / "test.jsonl"
            output_meta = tmp_path / "out.jsonl"
            write_jsonl([{"audio": str(audio), "text": "x"}], input_meta)
            candidate_text = tmp_path / "candidate_text.npy"
            test_text = tmp_path / "test_text.npy"
            np.save(candidate_text, np.ones((2, 2), dtype=np.float32))
            np.save(test_text, np.ones((1, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "candidate_text_embeddings"):
                retrieve_to_manifest(
                    method="ticl",
                    input_meta=str(input_meta),
                    output_meta=str(output_meta),
                    candidate_meta=str(input_meta),
                    candidate_text_embeddings_path=str(candidate_text),
                    test_text_embeddings_path=str(test_text),
                )

    def test_retrieve_to_manifest_writes_ice(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_a = tmp_path / "a.wav"
            audio_b = tmp_path / "b.wav"
            audio_a.write_bytes(b"placeholder")
            audio_b.write_bytes(b"placeholder")
            test_meta = tmp_path / "test.jsonl"
            candidate_meta = tmp_path / "candidate.jsonl"
            output_meta = tmp_path / "out.jsonl"
            write_jsonl([{"audio": str(audio_a), "text": "query"}], test_meta)
            write_jsonl([
                {"audio": str(audio_a), "text": "self"},
                {"audio": str(audio_b), "text": "other"},
            ], candidate_meta)
            candidate_text = tmp_path / "candidate_text.npy"
            test_text = tmp_path / "test_text.npy"
            np.save(candidate_text, np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))
            np.save(test_text, np.array([[1.0, 0.0]], dtype=np.float32))
            rows = retrieve_to_manifest(
                method="ticl",
                input_meta=str(test_meta),
                output_meta=str(output_meta),
                candidate_meta=str(candidate_meta),
                candidate_text_embeddings_path=str(candidate_text),
                test_text_embeddings_path=str(test_text),
                topk=1,
            )
            self.assertEqual(rows[0]["in_context_examples"][0]["text"], "other")
            loaded = load_manifest(output_meta, must_exist=False)
            self.assertEqual(loaded[0]["in_context_examples"][0]["text"], "other")

    def test_retrieve_to_manifest_can_include_ids_scores_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_a = tmp_path / "a.wav"
            audio_b = tmp_path / "b.wav"
            audio_a.write_bytes(b"placeholder")
            audio_b.write_bytes(b"placeholder")
            test_meta = tmp_path / "test.jsonl"
            candidate_meta = tmp_path / "candidate.jsonl"
            output_meta = tmp_path / "out.jsonl"
            write_jsonl([{"audio": str(audio_a), "text": "query"}], test_meta)
            write_jsonl([{"audio": str(audio_b), "text": "other"}], candidate_meta)
            candidate_text = tmp_path / "candidate_text.npy"
            test_text = tmp_path / "test_text.npy"
            np.save(candidate_text, np.array([[1.0, 0.0]], dtype=np.float32))
            np.save(test_text, np.array([[1.0, 0.0]], dtype=np.float32))
            rows = retrieve_to_manifest(
                method="ticl",
                input_meta=str(test_meta),
                output_meta=str(output_meta),
                candidate_meta=str(candidate_meta),
                candidate_text_embeddings_path=str(candidate_text),
                test_text_embeddings_path=str(test_text),
                topk=1,
                include_ids=True,
                include_scores=True,
                include_config=True,
                text_encoder_model_name="toy-text",
                preset="english",
            )
            row = rows[0]
            self.assertEqual(row["in_context_example_ids"], [0])
            self.assertAlmostEqual(row["in_context_example_scores"][0], 1.0, places=6)
            self.assertEqual(row["sicl_retriever_config"]["method"], "ticl")
            self.assertEqual(row["sicl_retriever_config"]["text_encoder_model_name"], "toy-text")
            self.assertIn("sha256", row["sicl_retriever_config"]["embedding_files"]["candidate_text"])

    def test_cli_accepts_legacy_embedding_aliases(self):
        args = build_parser().parse_args([
            "retrieve",
            "--input-meta", "test.jsonl",
            "--output-meta", "out.jsonl",
            "--path_to_candidate_text_embedding", "candidate.npy",
            "--path_to_test_text_embeddings", "test.npy",
        ])
        self.assertEqual(args.candidate_text_embeddings, "candidate.npy")
        self.assertEqual(args.test_text_embeddings, "test.npy")

    def test_minimal_ticl_example_cli(self):
        repo = Path(__file__).resolve().parents[1]
        example = repo / "examples" / "minimal"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            rc = main([
                "retrieve",
                "--method", "ticl",
                "--input-meta", str(example / "test.jsonl"),
                "--candidate-meta", str(example / "candidates.jsonl"),
                "--output-meta", str(output),
                "--candidate-text-embeddings", str(example / "candidate_text.npy"),
                "--test-text-embeddings", str(example / "test_text.npy"),
                "--topk", "2",
                "--include-ids",
            ])
            self.assertEqual(rc, 0)
            rows = read_jsonl(output)
            self.assertEqual(rows[0]["in_context_example_ids"], [0, 2])

    def test_minimal_ticl_plus_example_cli(self):
        repo = Path(__file__).resolve().parents[1]
        example = repo / "examples" / "minimal"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            rc = main([
                "retrieve",
                "--method", "ticl-plus",
                "--input-meta", str(example / "test.jsonl"),
                "--candidate-meta", str(example / "candidates.jsonl"),
                "--output-meta", str(output),
                "--candidate-text-embeddings", str(example / "candidate_text.npy"),
                "--test-text-embeddings", str(example / "test_text.npy"),
                "--candidate-audio-embeddings", str(example / "candidate_audio.npy"),
                "--test-audio-embeddings", str(example / "test_audio.npy"),
                "--topk", "2",
                "--include-ids",
            ])
            self.assertEqual(rc, 0)
            rows = read_jsonl(output)
            self.assertEqual(rows[0]["in_context_example_ids"], [2, 1])


if __name__ == "__main__":
    unittest.main()
