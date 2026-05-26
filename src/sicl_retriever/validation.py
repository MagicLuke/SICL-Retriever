from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .io import JsonRow, load_manifest


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_required_columns(rows: Sequence[JsonRow], columns: Sequence[str], label: str, report: ValidationReport) -> None:
    for idx, row in enumerate(rows):
        for column in columns:
            if column not in row:
                report.errors.append(f"{label} row {idx} is missing required column {column!r}")


def _check_embedding_rows(path: str | None, expected_rows: int, label: str, report: ValidationReport) -> None:
    if path is None:
        return
    if not Path(path).exists():
        report.errors.append(f"{label} embedding file does not exist: {path}")
        return
    try:
        embeddings = np.load(path, mmap_mode="r")
    except Exception as exc:
        report.errors.append(f"{label} embedding file could not be loaded: {path}: {exc}")
        return
    if embeddings.ndim != 2:
        report.errors.append(f"{label} embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[0] != expected_rows:
        report.errors.append(f"{label} embeddings have {embeddings.shape[0]} rows but manifest has {expected_rows} rows")


def validate_inputs(
    *,
    input_meta: str,
    candidate_meta: str | None = None,
    candidate_text_embeddings_path: str | None = None,
    test_text_embeddings_path: str | None = None,
    candidate_audio_embeddings_path: str | None = None,
    test_audio_embeddings_path: str | None = None,
    audio_column: str = "audio",
    required_columns: Sequence[str] = ("audio", "text"),
    allow_missing_audio: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    try:
        test_rows = load_manifest(input_meta, audio_column=audio_column, must_exist=not allow_missing_audio)
    except Exception as exc:
        report.errors.append(f"Could not load input manifest {input_meta}: {exc}")
        test_rows = []
    try:
        candidate_rows = load_manifest(candidate_meta or input_meta, audio_column=audio_column, must_exist=not allow_missing_audio)
    except Exception as exc:
        report.errors.append(f"Could not load candidate manifest {candidate_meta or input_meta}: {exc}")
        candidate_rows = []

    report.stats["test_rows"] = len(test_rows)
    report.stats["candidate_rows"] = len(candidate_rows)
    _check_required_columns(test_rows, required_columns, "input", report)
    _check_required_columns(candidate_rows, required_columns, "candidate", report)

    candidate_audio_values = [row.get(audio_column) for row in candidate_rows]
    duplicate_candidates = len(candidate_audio_values) - len(set(candidate_audio_values))
    if duplicate_candidates:
        report.warnings.append(f"Candidate manifest has {duplicate_candidates} duplicate {audio_column!r} values")
    report.stats["duplicate_candidate_audio"] = duplicate_candidates

    overlap = len(set(row.get(audio_column) for row in test_rows) & set(candidate_audio_values))
    report.stats["test_candidate_audio_overlap"] = overlap
    if overlap:
        report.warnings.append(f"Input and candidate manifests overlap on {overlap} {audio_column!r} values")

    _check_embedding_rows(candidate_text_embeddings_path, len(candidate_rows), "candidate text", report)
    _check_embedding_rows(test_text_embeddings_path, len(test_rows), "test text", report)
    _check_embedding_rows(candidate_audio_embeddings_path, len(candidate_rows), "candidate audio", report)
    _check_embedding_rows(test_audio_embeddings_path, len(test_rows), "test audio", report)
    return report


def summarize_retrieved_manifest(path: str, *, output_column: str = "in_context_examples", ids_column: str = "in_context_example_ids") -> dict[str, object]:
    rows = load_manifest(path, resolve_audio=False, must_exist=False)
    counts = []
    rows_with_ids = 0
    self_matches = 0
    for row in rows:
        examples = row.get(output_column, [])
        if isinstance(examples, list):
            counts.append(len(examples))
            row_audio = row.get("audio")
            self_matches += sum(1 for example in examples if isinstance(example, dict) and example.get("audio") == row_audio)
        else:
            counts.append(0)
        if ids_column in row:
            rows_with_ids += 1
    return {
        "rows": len(rows),
        "total_examples": sum(counts),
        "min_examples": min(counts) if counts else 0,
        "max_examples": max(counts) if counts else 0,
        "avg_examples": (sum(counts) / len(counts)) if counts else 0.0,
        "rows_with_ids": rows_with_ids,
        "self_matches": self_matches,
    }

