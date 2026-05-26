# CV-En TICL Demo

This example uses one real row from a prepared Common Voice English TICL manifest and small deterministic toy embeddings. It is intended for retrieval smoke tests, not benchmark reporting.

The JSONL files are deidentified to the minimum fields needed by this package: `audio` and `text`. They intentionally omit Common Voice client ids, demographic columns, split labels, source ranks, and internal source paths.

## Source Row

- Prepared TICL manifest: `/work/hdd/beiq/haolong2/MetaSICL/data/common_voice_cv-corpus-15.0-2023-09-08_en_dev/ticl_w_ice_from_common_voice_cv-corpus-15.0-2023-09-08_en_validated.jsonl`
- Query clip: `common_voice_en_19624951.mp3`
- Query text: `Because of facial deformity, she lives a life of fear and shame.`

The candidate set contains the first three retrieved examples from that prepared row:

1. `common_voice_en_20436647.mp3`
2. `common_voice_en_27143639.mp3`
3. `common_voice_en_19712824.mp3`

It also contains one deterministic distractor:

4. `common_voice_en_27710027.mp3`

The distractor is the first row in the local CV-en `validated.jsonl` that is not one of the selected positives.

## Run

```sh
sicl-retriever validate \
  --input-meta examples/cv_en/test.jsonl \
  --candidate-meta examples/cv_en/candidates.jsonl \
  --candidate-text-embeddings examples/cv_en/candidate_text.npy \
  --test-text-embeddings examples/cv_en/test_text.npy

sicl-retriever retrieve \
  --method ticl \
  --input-meta examples/cv_en/test.jsonl \
  --candidate-meta examples/cv_en/candidates.jsonl \
  --output-meta /tmp/cv_en_ticl_demo.jsonl \
  --candidate-text-embeddings examples/cv_en/candidate_text.npy \
  --test-text-embeddings examples/cv_en/test_text.npy \
  --topk 3 \
  --include-ids
```

The deterministic expected candidate ids are `[0, 1, 2]`.

## License And Redistribution Notice

The copied audio clips come from Mozilla Common Voice English, release `cv-corpus-15.0-2023-09-08`. Mozilla makes Common Voice datasets available under the Creative Commons CC0 public domain dedication unless otherwise specified. See https://commonvoice.mozilla.org/en/terms.

There is still a practical license/privacy concern with committing real voice clips: even CC0 speech data can carry dataset terms and contributor privacy expectations, including not trying to identify speakers. For a public release, review Mozilla's current Common Voice terms and consider replacing these MP3s with a small download script or synthetic audio if redistribution is not acceptable.
