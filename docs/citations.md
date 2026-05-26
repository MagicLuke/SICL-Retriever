# Citations

This repository supports the retrieval and preparation algorithms from two Speech In-Context Learning papers.

## TICL

TICL retrieves in-context examples by embedding a pseudo-transcript of the query audio and the reference transcripts in a candidate pool, then applying KNN search in text-embedding space.

```bibtex
@inproceedings{zheng2026ticl,
  title={TICL: Text-Embedding KNN For Speech In-Context Learning Unlocks Speech Recognition Abilities of Large Multimodal Models},
  author={Zheng, Haolong and Yegorova, Yekaterina and Hasegawa-Johnson, Mark},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  note={Accepted. Paper ID 9936},
  eprint={2509.13395},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2509.13395}
}
```

## TICL+

TICL+ extends TICL by taking the text-retrieved candidate set and reranking it with acoustic similarity from audio embeddings.

```bibtex
@inproceedings{zheng2025ticlplus,
  title={TICL+: A Case Study On Speech In-Context Learning for Children's Speech Recognition},
  author={Zheng, Haolong and Yegorova, Yekaterina and Hasegawa-Johnson, Mark},
  booktitle={IEEE ASRU 2025 Satellite Workshop-AI for Children's Speech and Language},
  year={2025},
  doi={10.48550/arXiv.2512.18263}
}
```

