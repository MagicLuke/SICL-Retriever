from __future__ import annotations

from typing import Sequence

from .io import load_manifest, write_jsonl


class WhisperASRPipeline:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        *,
        device: str | None = None,
        dtype=None,
        attn_implementation: str | None = None,
        max_new_tokens: int = 128,
        language: str = "en",
    ):
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise ImportError("pseudo-label generation requires torch and transformers; install sicl-retriever[prep]") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if "cuda" in str(self.device) else torch.float32)
        self.sampling_rate = 16000
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.processor = AutoProcessor.from_pretrained(model_name)
        kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **kwargs).to(self.device).eval()

    def _load_audio(self, path: str):
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("audio loading requires librosa; install sicl-retriever[prep]") from exc
        return librosa.load(path, sr=self.sampling_rate)[0]

    def transcribe(
        self,
        audio_paths: Sequence[str],
        *,
        batch_size: int = 64,
        show_progress: bool = True,
        task: str = "transcribe",
    ) -> list[str]:
        import os
        from tqdm import tqdm

        torch = self.torch
        paths = list(audio_paths)
        outputs: list[str] = []

        def collate_fn(batch_paths):
            audios = []
            for path in batch_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Audio file not found: {path}")
                audios.append(self._load_audio(path))
            return self.processor(audios, sampling_rate=self.sampling_rate, return_tensors="pt")

        dataloader = torch.utils.data.DataLoader(
            paths,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=("cuda" in str(self.device)),
            collate_fn=collate_fn,
        )
        iterator = tqdm(dataloader, desc=f"Transcribing on {self.device}") if show_progress else dataloader
        with torch.no_grad():
            for batch in iterator:
                input_features = batch["input_features"].to(self.device, dtype=self.dtype)
                generated_ids = self.model.generate(
                    input_features=input_features,
                    max_new_tokens=self.max_new_tokens,
                    task=task,
                    language=self.language,
                )
                outputs.extend(self.processor.batch_decode(generated_ids, skip_special_tokens=True))
        return outputs


def pseudo_label_manifest(
    input_jsonl: str,
    output_jsonl: str,
    *,
    input_column: str = "audio",
    output_column: str = "pseudo_label",
    model_name: str = "openai/whisper-large-v3-turbo",
    batch_size: int = 64,
    device: str | None = None,
    language: str = "en",
    max_new_tokens: int = 128,
) -> list[dict[str, object]]:
    rows = load_manifest(input_jsonl, audio_column=input_column)
    paths = [str(row[input_column]) for row in rows]
    pipeline = WhisperASRPipeline(
        model_name=model_name,
        device=device,
        language=language,
        max_new_tokens=max_new_tokens,
    )
    labels = pipeline.transcribe(paths, batch_size=batch_size)
    if len(labels) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} transcriptions, got {len(labels)}")
    output_rows = []
    for row, label in zip(rows, labels):
        copied = dict(row)
        copied[output_column] = label
        output_rows.append(copied)
    write_jsonl(output_rows, output_jsonl)
    return output_rows
