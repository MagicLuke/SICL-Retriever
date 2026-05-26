from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .asr import pseudo_label_manifest
from .embeddings import audio_embed_manifest, text_embed_manifest
from .filtering import filter_manifest
from .retrieval import retrieve_to_manifest


@dataclass(slots=True)
class PrepareConfig:
    input_meta: str
    output_meta: str
    work_dir: str
    candidate_meta: str | None = None
    method: str = "ticl"
    topk: int = 5
    batch_size: int = 64
    audio_column: str = "audio"
    text_column: str = "text"
    pseudo_label_column: str = "pseudo_label"
    min_duration: float = 1.0
    max_duration: float = 15.0
    asr_model_name: str = "openai/whisper-large-v3-turbo"
    text_encoder_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    audio_encoder_model_name: str = "openai/whisper-large-v3-turbo"
    pooling: str = "mean_only"
    device: str = "cpu"
    metric: str = "IP"
    language: str = "en"
    candidate_multiplier: int = 10
    overwrite: bool = False


def _should_run(path: Path, overwrite: bool) -> bool:
    return overwrite or not path.exists()


def run_prepare(config: PrepareConfig) -> list[dict[str, object]]:
    """Run the full filtering, pseudo-label, embedding, and retrieval pipeline."""
    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    method = config.method.replace("_", "-").lower()
    candidate_meta = config.candidate_meta or config.input_meta

    filtered_dir = work_dir / "filtered"
    pseudo_dir = work_dir / "pseudo_labels"
    emb_dir = work_dir / "embeddings"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    test_filtered = filtered_dir / "test.jsonl"
    candidate_filtered = filtered_dir / "candidate.jsonl"
    test_pseudo = pseudo_dir / "test.jsonl"

    if _should_run(test_filtered, config.overwrite):
        filter_manifest(
            config.input_meta,
            str(test_filtered),
            input_column=config.audio_column,
            min_duration=config.min_duration,
            max_duration=config.max_duration,
        )
    if _should_run(candidate_filtered, config.overwrite):
        filter_manifest(
            candidate_meta,
            str(candidate_filtered),
            input_column=config.audio_column,
            min_duration=config.min_duration,
            max_duration=config.max_duration,
        )
    if _should_run(test_pseudo, config.overwrite):
        pseudo_label_manifest(
            str(test_filtered),
            str(test_pseudo),
            input_column=config.audio_column,
            output_column=config.pseudo_label_column,
            model_name=config.asr_model_name,
            batch_size=config.batch_size,
            device=config.device,
            language=config.language,
        )

    candidate_text = emb_dir / "candidate_text.npy"
    test_text = emb_dir / "test_text.npy"
    candidate_audio = emb_dir / "candidate_audio.npy"
    test_audio = emb_dir / "test_audio.npy"

    if _should_run(candidate_text, config.overwrite):
        text_embed_manifest(
            str(candidate_filtered),
            str(candidate_text),
            input_column=config.text_column,
            model_name=config.text_encoder_model_name,
            batch_size=config.batch_size,
            device=config.device,
        )
    if _should_run(test_text, config.overwrite):
        text_embed_manifest(
            str(test_pseudo),
            str(test_text),
            input_column=config.pseudo_label_column,
            model_name=config.text_encoder_model_name,
            batch_size=config.batch_size,
            device=config.device,
        )

    candidate_audio_path = None
    test_audio_path = None
    if method == "ticl-plus":
        candidate_audio_path = str(candidate_audio)
        test_audio_path = str(test_audio)
        if _should_run(candidate_audio, config.overwrite):
            audio_embed_manifest(
                str(candidate_filtered),
                str(candidate_audio),
                input_column=config.audio_column,
                model_name=config.audio_encoder_model_name,
                pooling=config.pooling,
                batch_size=config.batch_size,
                device=config.device,
            )
        if _should_run(test_audio, config.overwrite):
            audio_embed_manifest(
                str(test_filtered),
                str(test_audio),
                input_column=config.audio_column,
                model_name=config.audio_encoder_model_name,
                pooling=config.pooling,
                batch_size=config.batch_size,
                device=config.device,
            )

    return retrieve_to_manifest(
        method=method,
        input_meta=str(test_pseudo),
        output_meta=config.output_meta,
        candidate_meta=str(candidate_filtered),
        candidate_text_embeddings_path=str(candidate_text),
        test_text_embeddings_path=str(test_text),
        candidate_audio_embeddings_path=candidate_audio_path,
        test_audio_embeddings_path=test_audio_path,
        topk=config.topk,
        batch_size=config.batch_size,
        metric=config.metric,
        device=config.device,
        candidate_multiplier=config.candidate_multiplier,
        audio_column=config.audio_column,
    )
