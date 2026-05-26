from __future__ import annotations

from typing import Iterable

from .io import JsonRow, load_manifest, write_jsonl


def duration_seconds(audio_path: str) -> float:
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError("duration filtering requires torchaudio; install sicl-retriever[prep]") from exc
    info = torchaudio.info(audio_path)
    return float(info.num_frames) / float(info.sample_rate)


def keep_by_duration(audio_path: str, *, min_duration: float = 1.0, max_duration: float = 15.0) -> bool:
    duration = duration_seconds(audio_path)
    return min_duration <= duration <= max_duration


def filter_rows_by_duration(
    rows: Iterable[JsonRow],
    *,
    input_column: str = "audio",
    min_duration: float = 1.0,
    max_duration: float = 15.0,
) -> list[JsonRow]:
    kept: list[JsonRow] = []
    for idx, row in enumerate(rows):
        value = row.get(input_column)
        if not isinstance(value, str):
            raise ValueError(f"Row {idx} column {input_column!r} must be an audio path string")
        if keep_by_duration(value, min_duration=min_duration, max_duration=max_duration):
            kept.append(dict(row))
    return kept


def filter_manifest(
    input_jsonl: str,
    output_jsonl: str,
    *,
    input_column: str = "audio",
    min_duration: float = 1.0,
    max_duration: float = 15.0,
) -> list[JsonRow]:
    rows = load_manifest(input_jsonl, audio_column=input_column)
    filtered = filter_rows_by_duration(
        rows,
        input_column=input_column,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    write_jsonl(filtered, output_jsonl)
    return filtered
