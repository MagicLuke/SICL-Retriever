from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

JsonRow = dict[str, object]


def read_jsonl(path: str | os.PathLike[str]) -> list[JsonRow]:
    """Read a JSON or JSONL manifest into a list of rows."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON array in {path}")
            return data
        rows = []
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object rows in {path}; line {line_no} is {type(row).__name__}")
            rows.append(row)
        return rows


def write_jsonl(rows: Iterable[JsonRow], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(path: str, root: str | os.PathLike[str] | None = None, *, must_exist: bool = True) -> str:
    """Resolve absolute, relative, and accidentally-rooted manifest paths."""
    raw = Path(path)
    candidates: list[Path] = []
    if raw.exists():
        return str(raw)
    if root is not None:
        root_path = Path(root)
        candidates.append(root_path / path.lstrip(os.sep))
        candidates.append(root_path / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if must_exist:
        tried = [str(raw)] + [str(candidate) for candidate in candidates]
        raise FileNotFoundError(f"Audio file not found. Tried: {', '.join(tried)}")
    return str(candidates[0] if candidates else raw)


def _resolve_value(value: object, root: Path, *, must_exist: bool) -> object:
    if isinstance(value, str):
        return resolve_path(value, root, must_exist=must_exist)
    if isinstance(value, list):
        return [_resolve_value(item, root, must_exist=must_exist) for item in value]
    if isinstance(value, dict) and "audio" in value:
        copied = dict(value)
        copied["audio"] = _resolve_value(copied["audio"], root, must_exist=must_exist)
        return copied
    return value


def resolve_audio_paths(
    rows: Sequence[JsonRow],
    root: str | os.PathLike[str],
    *,
    audio_column: str = "audio",
    must_exist: bool = True,
) -> list[JsonRow]:
    resolved: list[JsonRow] = []
    root_path = Path(root)
    for idx, row in enumerate(rows):
        if audio_column not in row:
            raise KeyError(f"Row {idx} is missing audio column {audio_column!r}")
        copied = dict(row)
        copied[audio_column] = _resolve_value(copied[audio_column], root_path, must_exist=must_exist)
        resolved.append(copied)
    return resolved


def load_manifest(
    path: str | os.PathLike[str],
    *,
    audio_column: str = "audio",
    resolve_audio: bool = True,
    must_exist: bool = True,
) -> list[JsonRow]:
    rows = read_jsonl(path)
    if not resolve_audio:
        return rows
    return resolve_audio_paths(rows, Path(path).parent, audio_column=audio_column, must_exist=must_exist)


def rows_from_column_dict(columns: dict[str, Sequence[object]]) -> list[JsonRow]:
    keys = list(columns.keys())
    if not keys:
        return []
    size = len(columns[keys[0]])
    for key in keys[1:]:
        if len(columns[key]) != size:
            raise ValueError("All column lists must have the same length")
    return [dict(zip(keys, values)) for values in zip(*(columns[key] for key in keys))]
