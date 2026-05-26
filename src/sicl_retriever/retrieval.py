from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import warnings
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from tqdm import tqdm

from .io import JsonRow, load_manifest, write_jsonl

Metric = Literal["IP", "L2"]


def _as_float32_matrix(name: str, value: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D embedding matrix, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return np.ascontiguousarray(array)


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return array / norms


def _sort_scores_and_indices(scores: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_scores = np.empty_like(scores)
    sorted_indices = np.empty_like(indices, dtype=np.int64)
    for row_idx in range(scores.shape[0]):
        valid = indices[row_idx] >= 0
        row_scores = scores[row_idx]
        row_indices = indices[row_idx].astype(np.int64)
        order = np.lexsort((row_indices[valid], -row_scores[valid]))
        valid_positions = np.nonzero(valid)[0][order]
        invalid_positions = np.nonzero(~valid)[0]
        full_order = np.concatenate([valid_positions, invalid_positions])
        sorted_scores[row_idx] = row_scores[full_order]
        sorted_indices[row_idx] = row_indices[full_order]
    return sorted_scores, sorted_indices


def _topk_by_scores(scores: np.ndarray, candidate_ids: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    top_scores: list[np.ndarray] = []
    top_ids: list[np.ndarray] = []
    for row_scores, row_ids in zip(scores, candidate_ids):
        order = np.lexsort((row_ids.astype(np.int64), -row_scores))[:topk]
        top_scores.append(row_scores[order])
        top_ids.append(row_ids[order])
    return np.vstack(top_scores).astype(np.float32), np.vstack(top_ids).astype(np.int64)


def _file_sha256(path: str | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _NumpyExactIndex:
    def __init__(self, embeddings: np.ndarray, metric: Metric):
        self.embeddings = _normalize_rows(embeddings) if metric == "IP" else embeddings
        self.metric = metric

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.embeddings.shape[0])
        if self.metric == "IP":
            scores = _normalize_rows(queries) @ self.embeddings.T
        else:
            diff = queries[:, None, :] - self.embeddings[None, :, :]
            scores = -np.sum(diff * diff, axis=-1)
        candidate_ids = np.broadcast_to(np.arange(self.embeddings.shape[0], dtype=np.int64), scores.shape)
        top_scores, top_ids = _topk_by_scores(scores, candidate_ids, k)
        return top_scores.astype(np.float32), top_ids.astype(np.int64)


class EmbeddingIndex:
    """Exact embedding index using FAISS when available, with a NumPy fallback."""

    def __init__(self, embeddings: np.ndarray, *, metric: Metric = "IP", device: str = "cpu"):
        self.metric = _normalize_metric(metric)
        self.device = device
        self.embeddings = _as_float32_matrix("embeddings", embeddings)
        self._index = self._build_index()

    def _build_index(self):
        try:
            import faiss  # type: ignore
        except ImportError:
            warnings.warn("faiss is not installed; using NumPy exact search fallback", RuntimeWarning, stacklevel=2)
            return _NumpyExactIndex(self.embeddings, self.metric)

        xb = self.embeddings.copy()
        dim = xb.shape[1]
        if self.metric == "IP":
            faiss.normalize_L2(xb)
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(xb)
        if self.device != "cpu":
            if hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    return faiss.index_cpu_to_gpu(res, 0, index)
                except Exception as exc:
                    warnings.warn(f"Could not move FAISS index to GPU; using CPU index: {exc}", RuntimeWarning, stacklevel=2)
            else:
                warnings.warn("Installed faiss package has no GPU support; using CPU index", RuntimeWarning, stacklevel=2)
        return index

    def search(self, queries: np.ndarray, k: int, *, batch_size: int = 64, desc: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        queries = _as_float32_matrix("queries", queries)
        if queries.shape[1] != self.embeddings.shape[1]:
            raise ValueError(f"Query dim {queries.shape[1]} does not match index dim {self.embeddings.shape[1]}")
        k = min(k, self.embeddings.shape[0])
        all_scores: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        iterator = range(0, queries.shape[0], batch_size)
        if desc:
            iterator = tqdm(iterator, desc=desc)
        for start in iterator:
            chunk = queries[start : start + batch_size].copy()
            if self.metric == "IP" and not isinstance(self._index, _NumpyExactIndex):
                import faiss  # type: ignore
                faiss.normalize_L2(chunk)
            scores, indices = self._index.search(chunk, k)
            scores, indices = _sort_scores_and_indices(np.asarray(scores), np.asarray(indices, dtype=np.int64))
            all_scores.append(np.asarray(scores))
            all_indices.append(np.asarray(indices, dtype=np.int64))
        return np.vstack(all_scores), np.vstack(all_indices)


def _normalize_metric(metric: str) -> Metric:
    metric = metric.upper()
    if metric not in {"IP", "L2"}:
        raise ValueError("metric must be 'IP' or 'L2'")
    return metric  # type: ignore[return-value]


@dataclass(slots=True)
class TICLRetriever:
    candidate_text_embeddings: np.ndarray
    metric: Metric = "IP"
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.metric = _normalize_metric(self.metric)
        self.candidate_text_embeddings = _as_float32_matrix("candidate_text_embeddings", self.candidate_text_embeddings)

    def retrieve(self, test_text_embeddings: np.ndarray, *, topk: int = 5, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        if topk <= 0:
            raise ValueError("topk must be positive")
        queries = _as_float32_matrix("test_text_embeddings", test_text_embeddings)
        index = EmbeddingIndex(self.candidate_text_embeddings, metric=self.metric, device=self.device)
        scores, ids = index.search(queries, topk, batch_size=batch_size, desc="Retrieving ICE via text embeddings")
        return scores, ids

    def retrieve_ids(self, test_text_embeddings: np.ndarray, *, topk: int = 5, batch_size: int = 64) -> np.ndarray:
        _, ids = self.retrieve(test_text_embeddings, topk=topk, batch_size=batch_size)
        return ids


@dataclass(slots=True)
class TICLPlusRetriever:
    candidate_audio_embeddings: np.ndarray
    candidate_text_embeddings: np.ndarray
    metric: Metric = "IP"
    device: str = "cpu"
    candidate_multiplier: int = 10

    def __post_init__(self) -> None:
        self.metric = _normalize_metric(self.metric)
        self.candidate_audio_embeddings = _as_float32_matrix("candidate_audio_embeddings", self.candidate_audio_embeddings)
        self.candidate_text_embeddings = _as_float32_matrix("candidate_text_embeddings", self.candidate_text_embeddings)
        if self.candidate_audio_embeddings.shape[0] != self.candidate_text_embeddings.shape[0]:
            raise ValueError("candidate audio/text embedding row counts must match")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")

    def retrieve(
        self,
        test_text_embeddings: np.ndarray,
        test_audio_embeddings: np.ndarray,
        *,
        topk: int = 5,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        if topk <= 0:
            raise ValueError("topk must be positive")
        text_queries = _as_float32_matrix("test_text_embeddings", test_text_embeddings)
        audio_queries = _as_float32_matrix("test_audio_embeddings", test_audio_embeddings)
        if text_queries.shape[0] != audio_queries.shape[0]:
            raise ValueError("test audio/text embedding row counts must match")
        initial_k = min(topk * self.candidate_multiplier, self.candidate_text_embeddings.shape[0])
        initial_ids = TICLRetriever(
            self.candidate_text_embeddings,
            metric=self.metric,
            device=self.device,
        ).retrieve_ids(text_queries, topk=initial_k, batch_size=batch_size)
        return rerank_by_embeddings(
            initial_ids,
            self.candidate_audio_embeddings,
            audio_queries,
            topk=topk,
            metric=self.metric,
            device=self.device,
            batch_size=batch_size,
        )

    def retrieve_ids(
        self,
        test_text_embeddings: np.ndarray,
        test_audio_embeddings: np.ndarray,
        *,
        topk: int = 5,
        batch_size: int = 64,
    ) -> np.ndarray:
        _, ids = self.retrieve(test_text_embeddings, test_audio_embeddings, topk=topk, batch_size=batch_size)
        return ids


def rerank_by_embeddings(
    candidate_ids: np.ndarray,
    candidate_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    topk: int,
    metric: Metric = "IP",
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    metric = _normalize_metric(metric)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if ids.ndim != 2:
        raise ValueError(f"candidate_ids must be 2D, got shape {ids.shape}")
    candidates = _as_float32_matrix("candidate_embeddings", candidate_embeddings)
    queries = _as_float32_matrix("query_embeddings", query_embeddings)
    if ids.shape[0] != queries.shape[0]:
        raise ValueError("candidate_ids row count must match query embedding row count")
    if candidates.shape[1] != queries.shape[1]:
        raise ValueError("candidate/query embedding dims must match")
    topk = min(topk, ids.shape[1])
    if topk <= 0:
        raise ValueError("topk must be positive")

    try:
        import torch
    except ImportError:
        return _rerank_numpy(ids, candidates, queries, topk=topk, metric=metric, batch_size=batch_size)

    if device != "cpu" and not torch.cuda.is_available():
        warnings.warn("CUDA requested for rerank but unavailable; using CPU", RuntimeWarning, stacklevel=2)
        device = "cpu"
    score_output: list[np.ndarray] = []
    id_output: list[np.ndarray] = []
    for start in tqdm(range(0, queries.shape[0], batch_size), desc="Reranking ICE via audio embeddings"):
        end = min(start + batch_size, queries.shape[0])
        ids_np = ids[start:end]
        query = torch.from_numpy(queries[start:end]).to(device)
        cand = torch.from_numpy(candidates[ids_np]).to(device)
        if metric == "IP":
            query = torch.nn.functional.normalize(query, dim=-1)
            cand = torch.nn.functional.normalize(cand, dim=-1)
            sims = torch.einsum("bd,bkd->bk", query, cand)
        else:
            sims = -((cand - query.unsqueeze(1)) ** 2).sum(dim=-1)
        top_scores, top_ids = _topk_by_scores(sims.cpu().numpy(), ids_np, topk)
        score_output.append(top_scores)
        id_output.append(top_ids)
    return np.vstack(score_output).astype(np.float32), np.vstack(id_output).astype(np.int64)


def rerank_ids_by_embeddings(
    candidate_ids: np.ndarray,
    candidate_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    topk: int,
    metric: Metric = "IP",
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    _, ids = rerank_by_embeddings(
        candidate_ids,
        candidate_embeddings,
        query_embeddings,
        topk=topk,
        metric=metric,
        device=device,
        batch_size=batch_size,
    )
    return ids


def _rerank_numpy(
    ids: np.ndarray,
    candidates: np.ndarray,
    queries: np.ndarray,
    *,
    topk: int,
    metric: Metric,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_output: list[np.ndarray] = []
    id_output: list[np.ndarray] = []
    for start in range(0, queries.shape[0], batch_size):
        end = min(start + batch_size, queries.shape[0])
        ids_np = ids[start:end]
        query = queries[start:end]
        cand = candidates[ids_np]
        if metric == "IP":
            query = _normalize_rows(query)
            cand = cand / np.maximum(np.linalg.norm(cand, axis=-1, keepdims=True), 1e-12)
            sims = np.einsum("bd,bkd->bk", query, cand)
        else:
            sims = -np.sum((cand - query[:, None, :]) ** 2, axis=-1)
        top_scores, top_ids = _topk_by_scores(sims, ids_np, topk)
        score_output.append(top_scores)
        id_output.append(top_ids)
    return np.vstack(score_output).astype(np.float32), np.vstack(id_output).astype(np.int64)


def attach_ice(
    test_rows: Sequence[JsonRow],
    candidate_rows: Sequence[JsonRow],
    retrieved_ids: np.ndarray,
    *,
    topk: int,
    retrieved_scores: np.ndarray | None = None,
    audio_column: str = "audio",
    output_column: str = "in_context_examples",
    ids_column: str | None = None,
    scores_column: str | None = None,
    config_column: str | None = None,
    config: dict[str, object] | None = None,
) -> list[JsonRow]:
    ids = np.asarray(retrieved_ids, dtype=np.int64)
    if ids.ndim != 2:
        raise ValueError(f"retrieved_ids must be 2D, got shape {ids.shape}")
    if ids.shape[0] != len(test_rows):
        raise ValueError(f"retrieved id rows ({ids.shape[0]}) do not match test rows ({len(test_rows)})")
    scores = None
    if retrieved_scores is not None:
        scores = np.asarray(retrieved_scores, dtype=np.float32)
        if scores.shape != ids.shape:
            raise ValueError(f"retrieved_scores shape {scores.shape} must match retrieved_ids shape {ids.shape}")
    output: list[JsonRow] = []
    for row_idx, (row, row_ids) in enumerate(zip(test_rows, ids)):
        query_audio = row.get(audio_column)
        examples: list[JsonRow] = []
        example_ids: list[int] = []
        example_scores: list[float] = []
        row_scores = scores[row_idx] if scores is not None else [None] * len(row_ids)
        for raw_id, raw_score in zip(row_ids, row_scores):
            candidate_id = int(raw_id)
            if candidate_id < 0 or candidate_id >= len(candidate_rows):
                raise IndexError(f"Retrieved candidate id {candidate_id} out of range for row {row_idx}")
            candidate = candidate_rows[candidate_id]
            if query_audio is not None and candidate.get(audio_column) == query_audio:
                continue
            examples.append(dict(candidate))
            example_ids.append(candidate_id)
            if raw_score is not None:
                example_scores.append(float(raw_score))
            if len(examples) >= topk:
                break
        copied = dict(row)
        copied[output_column] = examples
        if ids_column is not None:
            copied[ids_column] = example_ids
        if scores_column is not None:
            copied[scores_column] = example_scores
        if config_column is not None and config is not None:
            copied[config_column] = config
        output.append(copied)
    return output


def validate_embedding_rows(name: str, embeddings: np.ndarray, rows: Sequence[JsonRow]) -> None:
    if embeddings.shape[0] != len(rows):
        raise ValueError(f"{name} has {embeddings.shape[0]} rows but manifest has {len(rows)} rows")


def build_retrieval_config(
    *,
    method: str,
    metric: str,
    topk: int,
    retrieval_topk: int,
    batch_size: int,
    device: str,
    candidate_multiplier: int,
    input_meta: str,
    candidate_meta: str | None,
    candidate_text_embeddings_path: str,
    test_text_embeddings_path: str,
    candidate_audio_embeddings_path: str | None,
    test_audio_embeddings_path: str | None,
    text_encoder_model_name: str | None = None,
    audio_encoder_model_name: str | None = None,
    asr_model_name: str | None = None,
    preset: str | None = None,
) -> dict[str, object]:
    embedding_files = {
        "candidate_text": {
            "path": candidate_text_embeddings_path,
            "sha256": _file_sha256(candidate_text_embeddings_path),
        },
        "test_text": {
            "path": test_text_embeddings_path,
            "sha256": _file_sha256(test_text_embeddings_path),
        },
    }
    if candidate_audio_embeddings_path is not None:
        embedding_files["candidate_audio"] = {
            "path": candidate_audio_embeddings_path,
            "sha256": _file_sha256(candidate_audio_embeddings_path),
        }
    if test_audio_embeddings_path is not None:
        embedding_files["test_audio"] = {
            "path": test_audio_embeddings_path,
            "sha256": _file_sha256(test_audio_embeddings_path),
        }
    return {
        "method": method,
        "metric": metric,
        "topk": topk,
        "retrieval_topk": retrieval_topk,
        "batch_size": batch_size,
        "device": device,
        "candidate_multiplier": candidate_multiplier,
        "input_meta": input_meta,
        "candidate_meta": candidate_meta or input_meta,
        "text_encoder_model_name": text_encoder_model_name,
        "audio_encoder_model_name": audio_encoder_model_name,
        "asr_model_name": asr_model_name,
        "preset": preset,
        "embedding_files": embedding_files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def retrieve_to_manifest(
    *,
    method: str,
    input_meta: str,
    output_meta: str,
    candidate_meta: str | None = None,
    candidate_text_embeddings_path: str,
    test_text_embeddings_path: str,
    candidate_audio_embeddings_path: str | None = None,
    test_audio_embeddings_path: str | None = None,
    topk: int = 5,
    batch_size: int = 64,
    metric: str = "IP",
    device: str = "cpu",
    candidate_multiplier: int = 10,
    audio_column: str = "audio",
    ids_column: str | None = None,
    scores_column: str | None = None,
    config_column: str | None = None,
    include_ids: bool = False,
    include_scores: bool = False,
    include_config: bool = False,
    text_encoder_model_name: str | None = None,
    audio_encoder_model_name: str | None = None,
    asr_model_name: str | None = None,
    preset: str | None = None,
) -> list[JsonRow]:
    method = method.replace("_", "-").lower()
    if method not in {"ticl", "ticl-plus"}:
        raise ValueError("method must be 'ticl' or 'ticl-plus'")
    test_rows = load_manifest(input_meta, audio_column=audio_column)
    candidate_rows = load_manifest(candidate_meta or input_meta, audio_column=audio_column)
    candidate_text = np.load(candidate_text_embeddings_path)
    test_text = np.load(test_text_embeddings_path)
    validate_embedding_rows("candidate_text_embeddings", candidate_text, candidate_rows)
    validate_embedding_rows("test_text_embeddings", test_text, test_rows)
    retrieval_topk = min(topk + 1, len(candidate_rows))
    if method == "ticl":
        scores, ids = TICLRetriever(candidate_text, metric=_normalize_metric(metric), device=device).retrieve(
            test_text,
            topk=retrieval_topk,
            batch_size=batch_size,
        )
    else:
        if not candidate_audio_embeddings_path or not test_audio_embeddings_path:
            raise ValueError("TICL+ requires candidate and test audio embeddings")
        candidate_audio = np.load(candidate_audio_embeddings_path)
        test_audio = np.load(test_audio_embeddings_path)
        validate_embedding_rows("candidate_audio_embeddings", candidate_audio, candidate_rows)
        validate_embedding_rows("test_audio_embeddings", test_audio, test_rows)
        scores, ids = TICLPlusRetriever(
            candidate_audio,
            candidate_text,
            metric=_normalize_metric(metric),
            device=device,
            candidate_multiplier=candidate_multiplier,
        ).retrieve(test_text, test_audio, topk=retrieval_topk, batch_size=batch_size)
    config = None
    if include_config or config_column is not None:
        config = build_retrieval_config(
            method=method,
            metric=_normalize_metric(metric),
            topk=topk,
            retrieval_topk=retrieval_topk,
            batch_size=batch_size,
            device=device,
            candidate_multiplier=candidate_multiplier,
            input_meta=input_meta,
            candidate_meta=candidate_meta,
            candidate_text_embeddings_path=candidate_text_embeddings_path,
            test_text_embeddings_path=test_text_embeddings_path,
            candidate_audio_embeddings_path=candidate_audio_embeddings_path,
            test_audio_embeddings_path=test_audio_embeddings_path,
            text_encoder_model_name=text_encoder_model_name,
            audio_encoder_model_name=audio_encoder_model_name,
            asr_model_name=asr_model_name,
            preset=preset,
        )
    output_rows = attach_ice(
        test_rows,
        candidate_rows,
        ids,
        topk=topk,
        retrieved_scores=scores if include_scores or scores_column is not None else None,
        audio_column=audio_column,
        ids_column=ids_column or ("in_context_example_ids" if include_ids else None),
        scores_column=scores_column or ("in_context_example_scores" if include_scores else None),
        config_column=config_column or ("sicl_retriever_config" if include_config else None),
        config=config,
    )
    write_jsonl(output_rows, output_meta)
    return output_rows
