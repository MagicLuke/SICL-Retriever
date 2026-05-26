from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PaperPreset:
    name: str
    method: str
    text_encoder_model_name: str
    audio_encoder_model_name: str
    asr_model_name: str
    candidate_multiplier: int
    pooling: str = "mean_only"
    metric: str = "IP"


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    method: str
    topk: int
    batch_size: int
    metric: str
    device: str
    candidate_multiplier: int
    text_encoder_model_name: str | None = None
    audio_encoder_model_name: str | None = None
    asr_model_name: str | None = None
    preset: str | None = None


PRESETS: dict[str, PaperPreset] = {
    "english": PaperPreset(
        name="english",
        method="ticl",
        text_encoder_model_name="sentence-transformers/all-mpnet-base-v2",
        audio_encoder_model_name="openai/whisper-large-v3-turbo",
        asr_model_name="openai/whisper-large-v3-turbo",
        candidate_multiplier=10,
    ),
    "multilingual": PaperPreset(
        name="multilingual",
        method="ticl",
        text_encoder_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        audio_encoder_model_name="openai/whisper-large-v3-turbo",
        asr_model_name="openai/whisper-large-v3-turbo",
        candidate_multiplier=10,
    ),
    "children": PaperPreset(
        name="children",
        method="ticl-plus",
        text_encoder_model_name="sentence-transformers/all-mpnet-base-v2",
        audio_encoder_model_name="openai/whisper-large-v3-turbo",
        asr_model_name="openai/whisper-large-v3-turbo",
        candidate_multiplier=10,
    ),
}


def get_preset(name: str | None) -> PaperPreset | None:
    if name is None:
        return None
    try:
        return PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Available presets: {available}") from exc

