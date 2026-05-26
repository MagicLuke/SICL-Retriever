from __future__ import annotations

import argparse
import json
import sys

from .asr import pseudo_label_manifest
from .embeddings import audio_embed_manifest, text_embed_manifest
from .filtering import filter_manifest
from .pipeline import PrepareConfig, run_prepare
from .presets import PRESETS, get_preset
from .retrieval import retrieve_to_manifest
from .validation import summarize_retrieved_manifest, validate_inputs


def _add_io_aliases(parser: argparse.ArgumentParser, *, meta_names: bool = False) -> None:
    if meta_names:
        parser.add_argument("--input-meta", "--input_meta", dest="input_meta", required=True)
        parser.add_argument("--output-meta", "--output_meta", dest="output_meta", required=True)
    else:
        parser.add_argument("--input-jsonl", "--input_jsonl", dest="input_jsonl", required=True)
        parser.add_argument("--output-jsonl", "--output_jsonl", dest="output_jsonl", required=True)


def _add_embedding_path_aliases(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidate-text-embeddings",
        "--candidate-text-embedding",
        "--path_to_candidate_text_embeddings",
        "--path_to_candidate_text_embedding",
        dest="candidate_text_embeddings",
        required=True,
    )
    parser.add_argument(
        "--test-text-embeddings",
        "--test-text-embedding",
        "--path_to_test_text_embeddings",
        "--path_to_test_text_embedding",
        dest="test_text_embeddings",
        required=True,
    )
    parser.add_argument(
        "--candidate-audio-embeddings",
        "--candidate-audio-embedding",
        "--path_to_candidate_audio_embeddings",
        "--path_to_candidate_audio_embedding",
        dest="candidate_audio_embeddings",
        default=None,
    )
    parser.add_argument(
        "--test-audio-embeddings",
        "--test-audio-embedding",
        "--path_to_test_audio_embeddings",
        "--path_to_test_audio_embedding",
        dest="test_audio_embeddings",
        default=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sicl-retriever", description="SICL retrieval and preparation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_filter = sub.add_parser("filter", help="Filter an audio manifest by duration")
    _add_io_aliases(p_filter)
    p_filter.add_argument("--input-column", "--input_column", dest="input_column", default="audio")
    p_filter.add_argument("--min-duration", "--min_duration", dest="min_duration", type=float, default=1.0)
    p_filter.add_argument("--max-duration", "--max_duration", dest="max_duration", type=float, default=15.0)
    p_filter.add_argument("--num-proc", "--num_proc", dest="num_proc", type=int, default=1, help="Accepted for compatibility; unused")

    p_asr = sub.add_parser("pseudo-label", help="Generate Whisper pseudo labels")
    _add_io_aliases(p_asr)
    p_asr.add_argument("--input-column", "--input_column", dest="input_column", default="audio")
    p_asr.add_argument("--output-column", "--output_column", dest="output_column", default="pseudo_label")
    p_asr.add_argument("--model-name", "--model_name", dest="model_name", default="openai/whisper-large-v3-turbo")
    p_asr.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    p_asr.add_argument("--device", default=None)
    p_asr.add_argument("--language", default="en")
    p_asr.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=128)

    p_text = sub.add_parser("text-embed", help="Extract text embeddings")
    p_text.add_argument("--input-jsonl", "--input_jsonl", dest="input_jsonl", required=True)
    p_text.add_argument("--output-npy", "--output_npy", dest="output_npy", required=True)
    p_text.add_argument("--input-column", "--input_column", dest="input_column", default="text")
    p_text.add_argument("--model-name", "--model_name", dest="model_name", default="sentence-transformers/all-mpnet-base-v2")
    p_text.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    p_text.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=1000)
    p_text.add_argument("--device", default=None)
    p_text.add_argument("--num-proc", "--num_proc", dest="num_proc", type=int, default=1, help="Accepted for compatibility; unused")

    p_audio = sub.add_parser("audio-embed", help="Extract Whisper audio embeddings")
    p_audio.add_argument("--input-jsonl", "--input_jsonl", dest="input_jsonl", required=True)
    p_audio.add_argument("--output-npy", "--output_npy", dest="output_npy", required=True)
    p_audio.add_argument("--input-column", "--input_column", dest="input_column", default="audio")
    p_audio.add_argument("--model-name", "--model_name", dest="model_name", default="openai/whisper-large-v3-turbo")
    p_audio.add_argument("--pooling", default="mean_only")
    p_audio.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    p_audio.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    p_audio.add_argument("--device", default=None)

    p_retrieve = sub.add_parser("retrieve", help="Attach ICE examples to a manifest")
    p_retrieve.add_argument("--method", choices=["ticl", "ticl-plus", "ticl_plus"], default=None)
    p_retrieve.add_argument("--preset", choices=sorted(PRESETS), default=None)
    _add_io_aliases(p_retrieve, meta_names=True)
    p_retrieve.add_argument("--candidate-meta", "--candidate_meta", dest="candidate_meta", default=None)
    _add_embedding_path_aliases(p_retrieve)
    p_retrieve.add_argument("--topk", type=int, default=5)
    p_retrieve.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    p_retrieve.add_argument("--metric", default="IP")
    p_retrieve.add_argument("--device", default="cpu")
    p_retrieve.add_argument("--candidate-multiplier", "--candidate_multiplier", dest="candidate_multiplier", type=int, default=10)
    p_retrieve.add_argument("--audio-column", "--audio_column", dest="audio_column", default="audio")
    p_retrieve.add_argument("--ids-column", "--ids_column", dest="ids_column", default=None)
    p_retrieve.add_argument("--scores-column", "--scores_column", dest="scores_column", default=None)
    p_retrieve.add_argument("--config-column", "--config_column", dest="config_column", default=None)
    p_retrieve.add_argument("--include-ids", "--include_ids", dest="include_ids", action="store_true")
    p_retrieve.add_argument("--include-scores", "--include_scores", dest="include_scores", action="store_true")
    p_retrieve.add_argument("--include-config", "--include_config", dest="include_config", action="store_true")
    p_retrieve.add_argument("--text-encoder-model-name", "--text_encoder_model_name", dest="text_encoder_model_name", default=None)
    p_retrieve.add_argument("--audio-encoder-model-name", "--audio_encoder_model_name", dest="audio_encoder_model_name", default=None)
    p_retrieve.add_argument("--asr-model-name", "--asr_model_name", dest="asr_model_name", default=None)
    p_retrieve.add_argument("--num-proc", "--num_proc", dest="num_proc", type=int, default=1, help="Accepted for compatibility; unused")

    p_prepare = sub.add_parser("prepare", help="Run full filter/pseudo-label/embed/retrieve pipeline")
    p_prepare.add_argument("--method", choices=["ticl", "ticl-plus", "ticl_plus"], default=None)
    p_prepare.add_argument("--preset", choices=sorted(PRESETS), default=None)
    _add_io_aliases(p_prepare, meta_names=True)
    p_prepare.add_argument("--candidate-meta", "--candidate_meta", dest="candidate_meta", default=None)
    p_prepare.add_argument("--work-dir", "--work_dir", dest="work_dir", required=True)
    p_prepare.add_argument("--topk", type=int, default=5)
    p_prepare.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    p_prepare.add_argument("--audio-column", "--audio_column", dest="audio_column", default="audio")
    p_prepare.add_argument("--text-column", "--text_column", dest="text_column", default="text")
    p_prepare.add_argument("--pseudo-label-column", "--pseudo_label_column", dest="pseudo_label_column", default="pseudo_label")
    p_prepare.add_argument("--min-duration", "--min_duration", dest="min_duration", type=float, default=1.0)
    p_prepare.add_argument("--max-duration", "--max_duration", dest="max_duration", type=float, default=15.0)
    p_prepare.add_argument("--asr-model-name", "--asr_model_name", dest="asr_model_name", default=None)
    p_prepare.add_argument("--text-encoder-model-name", "--text_encoder_model_name", dest="text_encoder_model_name", default=None)
    p_prepare.add_argument("--audio-encoder-model-name", "--audio_encoder_model_name", dest="audio_encoder_model_name", default=None)
    p_prepare.add_argument("--pooling", default="mean_only")
    p_prepare.add_argument("--device", default="cpu")
    p_prepare.add_argument("--metric", default="IP")
    p_prepare.add_argument("--language", default="en")
    p_prepare.add_argument("--candidate-multiplier", "--candidate_multiplier", dest="candidate_multiplier", type=int, default=10)
    p_prepare.add_argument("--overwrite", action="store_true")

    p_validate = sub.add_parser("validate", help="Validate manifests and embedding row counts")
    p_validate.add_argument("--input-meta", "--input_meta", dest="input_meta", required=True)
    p_validate.add_argument("--candidate-meta", "--candidate_meta", dest="candidate_meta", default=None)
    p_validate.add_argument("--audio-column", "--audio_column", dest="audio_column", default="audio")
    p_validate.add_argument("--required-column", "--required_column", dest="required_columns", action="append", default=None)
    p_validate.add_argument("--allow-missing-audio", "--allow_missing_audio", dest="allow_missing_audio", action="store_true")
    p_validate.add_argument("--candidate-text-embeddings", "--path_to_candidate_text_embeddings", "--path_to_candidate_text_embedding", dest="candidate_text_embeddings", default=None)
    p_validate.add_argument("--test-text-embeddings", "--path_to_test_text_embeddings", "--path_to_test_text_embedding", dest="test_text_embeddings", default=None)
    p_validate.add_argument("--candidate-audio-embeddings", "--path_to_candidate_audio_embeddings", "--path_to_candidate_audio_embedding", dest="candidate_audio_embeddings", default=None)
    p_validate.add_argument("--test-audio-embeddings", "--path_to_test_audio_embeddings", "--path_to_test_audio_embedding", dest="test_audio_embeddings", default=None)

    p_summarize = sub.add_parser("summarize", help="Summarize a retrieved ICE manifest")
    p_summarize.add_argument("--input-meta", "--input_meta", dest="input_meta", required=True)
    p_summarize.add_argument("--output-column", "--output_column", dest="output_column", default="in_context_examples")
    p_summarize.add_argument("--ids-column", "--ids_column", dest="ids_column", default="in_context_example_ids")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "filter":
        filter_manifest(
            args.input_jsonl,
            args.output_jsonl,
            input_column=args.input_column,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    elif args.command == "pseudo-label":
        pseudo_label_manifest(
            args.input_jsonl,
            args.output_jsonl,
            input_column=args.input_column,
            output_column=args.output_column,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.command == "text-embed":
        text_embed_manifest(
            args.input_jsonl,
            args.output_npy,
            input_column=args.input_column,
            model_name=args.model_name,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            device=args.device,
        )
    elif args.command == "audio-embed":
        audio_embed_manifest(
            args.input_jsonl,
            args.output_npy,
            input_column=args.input_column,
            model_name=args.model_name,
            pooling=args.pooling,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
    elif args.command == "retrieve":
        preset = get_preset(args.preset)
        method = args.method or (preset.method if preset else "ticl")
        candidate_multiplier = args.candidate_multiplier
        if preset and candidate_multiplier == 10:
            candidate_multiplier = preset.candidate_multiplier
        retrieve_to_manifest(
            method=method,
            input_meta=args.input_meta,
            output_meta=args.output_meta,
            candidate_meta=args.candidate_meta,
            candidate_text_embeddings_path=args.candidate_text_embeddings,
            test_text_embeddings_path=args.test_text_embeddings,
            candidate_audio_embeddings_path=args.candidate_audio_embeddings,
            test_audio_embeddings_path=args.test_audio_embeddings,
            topk=args.topk,
            batch_size=args.batch_size,
            metric=args.metric,
            device=args.device,
            candidate_multiplier=candidate_multiplier,
            audio_column=args.audio_column,
            ids_column=args.ids_column,
            scores_column=args.scores_column,
            config_column=args.config_column,
            include_ids=args.include_ids,
            include_scores=args.include_scores,
            include_config=args.include_config,
            text_encoder_model_name=args.text_encoder_model_name or (preset.text_encoder_model_name if preset else None),
            audio_encoder_model_name=args.audio_encoder_model_name or (preset.audio_encoder_model_name if preset else None),
            asr_model_name=args.asr_model_name or (preset.asr_model_name if preset else None),
            preset=args.preset,
        )
    elif args.command == "prepare":
        preset = get_preset(args.preset)
        method = args.method or (preset.method if preset else "ticl")
        run_prepare(
            PrepareConfig(
                input_meta=args.input_meta,
                output_meta=args.output_meta,
                candidate_meta=args.candidate_meta,
                work_dir=args.work_dir,
                method=method,
                topk=args.topk,
                batch_size=args.batch_size,
                audio_column=args.audio_column,
                text_column=args.text_column,
                pseudo_label_column=args.pseudo_label_column,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                asr_model_name=args.asr_model_name or (preset.asr_model_name if preset else "openai/whisper-large-v3-turbo"),
                text_encoder_model_name=args.text_encoder_model_name or (preset.text_encoder_model_name if preset else "sentence-transformers/all-mpnet-base-v2"),
                audio_encoder_model_name=args.audio_encoder_model_name or (preset.audio_encoder_model_name if preset else "openai/whisper-large-v3-turbo"),
                pooling=args.pooling or (preset.pooling if preset else "mean_only"),
                device=args.device,
                metric=args.metric,
                language=args.language,
                candidate_multiplier=preset.candidate_multiplier if preset and args.candidate_multiplier == 10 else args.candidate_multiplier,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "validate":
        report = validate_inputs(
            input_meta=args.input_meta,
            candidate_meta=args.candidate_meta,
            candidate_text_embeddings_path=args.candidate_text_embeddings,
            test_text_embeddings_path=args.test_text_embeddings,
            candidate_audio_embeddings_path=args.candidate_audio_embeddings,
            test_audio_embeddings_path=args.test_audio_embeddings,
            audio_column=args.audio_column,
            required_columns=tuple(args.required_columns or ["audio", "text"]),
            allow_missing_audio=args.allow_missing_audio,
        )
        print(json.dumps({"ok": report.ok, "errors": report.errors, "warnings": report.warnings, "stats": report.stats}, indent=2))
        if not report.ok:
            return 1
    elif args.command == "summarize":
        summary = summarize_retrieved_manifest(args.input_meta, output_column=args.output_column, ids_column=args.ids_column)
        print(json.dumps(summary, indent=2))
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
