import tempfile
import unittest
from pathlib import Path

from sicl_retriever.io import load_manifest, write_jsonl


class IoTests(unittest.TestCase):
    def test_load_manifest_resolves_relative_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio" / "x.wav"
            audio.parent.mkdir()
            audio.write_bytes(b"placeholder")
            manifest = root / "manifest.jsonl"
            write_jsonl([{"audio": "audio/x.wav", "text": "x"}], manifest)
            rows = load_manifest(manifest)
            self.assertEqual(rows[0]["audio"], str(audio))

    def test_load_manifest_resolves_leading_slash_against_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio" / "x.wav"
            audio.parent.mkdir()
            audio.write_bytes(b"placeholder")
            manifest = root / "manifest.jsonl"
            write_jsonl([{"audio": "/audio/x.wav", "text": "x"}], manifest)
            rows = load_manifest(manifest)
            self.assertEqual(rows[0]["audio"], str(audio))


if __name__ == "__main__":
    unittest.main()
