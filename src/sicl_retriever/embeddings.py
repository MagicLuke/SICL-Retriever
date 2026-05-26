from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .io import load_manifest


def compute_stats_embeddings(embeddings, attention_mask=None, *, use_masking: bool = True, pooling_strategy: str = "mean_only"):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("audio embedding pooling requires torch; install sicl-retriever[prep]") from exc

    if use_masking and attention_mask is not None:
        feat_mask = attention_mask.unsqueeze(-1).float()
        if feat_mask.shape[1] != embeddings.shape[1]:
            feat_mask = torch.nn.functional.interpolate(
                feat_mask.transpose(1, 2),
                size=embeddings.shape[1],
                mode="nearest",
            ).transpose(1, 2)
        lengths = feat_mask.sum(dim=1, keepdim=True).clamp(min=1)
        if pooling_strategy == "mean_std":
            mean = (embeddings * feat_mask).sum(dim=1) / lengths.squeeze(1)
            var = ((embeddings - mean.unsqueeze(1)) ** 2 * feat_mask).sum(dim=1) / lengths.squeeze(1)
            return torch.cat([mean, var.sqrt()], dim=1)
        if pooling_strategy == "mean_only":
            return (embeddings * feat_mask).sum(dim=1) / lengths.squeeze(1)
        if pooling_strategy == "max_pool":
            masked = embeddings * feat_mask + (1 - feat_mask) * -1e9
            return masked.max(dim=1)[0]
        if pooling_strategy == "attention_pool":
            weights = torch.softmax(torch.sum(embeddings * feat_mask, dim=-1, keepdim=True), dim=1) * feat_mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            return (embeddings * weights).sum(dim=1)
    else:
        if pooling_strategy == "mean_std":
            return torch.cat([embeddings.mean(dim=1), embeddings.std(dim=1)], dim=1)
        if pooling_strategy == "mean_only":
            return embeddings.mean(dim=1)
        if pooling_strategy == "max_pool":
            return embeddings.max(dim=1)[0]
        if pooling_strategy == "attention_pool":
            weights = torch.softmax(torch.sum(embeddings, dim=-1, keepdim=True), dim=1)
            return (embeddings * weights).sum(dim=1)
    raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")


class SentenceTransformerEmbeddingExtractor:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("text embedding extraction requires sentence-transformers; install sicl-retriever[prep]") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 128,
        chunk_size: int = 1000,
        device: str | list[str] | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        import inspect

        kwargs = {
            "batch_size": batch_size,
            "device": device,
            "show_progress_bar": show_progress,
        }
        if "chunk_size" in inspect.signature(self.model.encode).parameters:
            kwargs["chunk_size"] = chunk_size
        embeddings = self.model.encode(list(texts), **kwargs)
        return np.asarray(embeddings, dtype=np.float32)


class WhisperEmbeddingExtractor:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        *,
        device: str | None = None,
        dtype=None,
        attn_implementation: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoProcessor, WhisperModel
        except ImportError as exc:
            raise ImportError("audio embedding extraction requires torch and transformers; install sicl-retriever[prep]") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if "cuda" in str(self.device) else torch.float32)
        self.sampling_rate = 16000
        self.processor = AutoProcessor.from_pretrained(model_name)
        kwargs = {"torch_dtype": self.dtype}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        elif "cuda" in str(self.device):
            kwargs["attn_implementation"] = "flash_attention_2"
        try:
            self.model = WhisperModel.from_pretrained(model_name, **kwargs).get_encoder().to(self.device).eval()
        except Exception:
            kwargs.pop("attn_implementation", None)
            self.model = WhisperModel.from_pretrained(model_name, **kwargs).get_encoder().to(self.device).eval()

    def _load_audio(self, path: str):
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("audio loading requires librosa; install sicl-retriever[prep]") from exc
        return librosa.load(path, sr=self.sampling_rate)[0]

    def encode(
        self,
        audio_paths: Sequence[str],
        *,
        pooling: str = "mean_only",
        batch_size: int = 64,
        num_workers: int = 4,
        show_progress: bool = True,
    ) -> np.ndarray:
        from tqdm import tqdm

        torch = self.torch
        paths = list(audio_paths)
        embeddings: list[np.ndarray] = []

        def collate_fn(batch_paths):
            audios = []
            for path in batch_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Audio file not found: {path}")
                audios.append(self._load_audio(path))
            return self.processor(
                audios,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )

        dataloader = torch.utils.data.DataLoader(
            paths,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        iterator = tqdm(dataloader, desc=f"Extracting Whisper embeddings on {self.device}") if show_progress else dataloader
        with torch.no_grad():
            for batch in iterator:
                batch = {key: value.to(self.device, dtype=self.dtype) if torch.is_tensor(value) else value for key, value in batch.items()}
                hidden = self.model(**batch).last_hidden_state
                pooled = compute_stats_embeddings(hidden, batch.get("attention_mask"), pooling_strategy=pooling).cpu()
                embeddings.extend(pooled.numpy())
        return np.asarray(embeddings, dtype=np.float32)


def save_embeddings(embeddings: np.ndarray, output_npy: str | os.PathLike[str]) -> None:
    output = Path(output_npy)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(embeddings, dtype=np.float32))


def text_embed_manifest(
    input_jsonl: str,
    output_npy: str,
    *,
    input_column: str = "text",
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 128,
    chunk_size: int = 1000,
    device: str | list[str] | None = None,
) -> np.ndarray:
    rows = load_manifest(input_jsonl, resolve_audio=False)
    texts = [str(row.get(input_column, "")) for row in rows]
    extractor = SentenceTransformerEmbeddingExtractor(model_name)
    embeddings = extractor.encode(texts, batch_size=batch_size, chunk_size=chunk_size, device=device)
    save_embeddings(embeddings, output_npy)
    return embeddings


def audio_embed_manifest(
    input_jsonl: str,
    output_npy: str,
    *,
    input_column: str = "audio",
    model_name: str = "openai/whisper-large-v3-turbo",
    pooling: str = "mean_only",
    batch_size: int = 64,
    num_workers: int = 4,
    device: str | None = None,
) -> np.ndarray:
    rows = load_manifest(input_jsonl, audio_column=input_column)
    paths = [str(row[input_column]) for row in rows]
    extractor = WhisperEmbeddingExtractor(model_name, device=device)
    embeddings = extractor.encode(paths, pooling=pooling, batch_size=batch_size, num_workers=num_workers)
    save_embeddings(embeddings, output_npy)
    return embeddings
