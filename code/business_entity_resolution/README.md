# Business Entity Resolution — Amazon ML Challenge 2026

Blocking + pairwise gradient-boosted matcher that links every Source-1 business to its
Source-2 / Source-3 records. CPU only, no external data or APIs, no pretrained models.

```
raw TSV ─► normalize (multi-representation, anyascii transliteration, legal-form class)
        ─► blocking: IDF-weighted inverted index, 6 key types, name / address / combined top-k
        ─► candidate_pairs.tsv
        ─► 54 pair features (rapidfuzz + token/number overlap + blocking context)
        ─► LightGBM P(match) ─► exclusivity (record → its best S1) ─► threshold (macro-F0.5 tuned)
        ─► matching_results.tsv ─► official validator
```

## Environment

```bash
python -m pip install -r requirements.txt      # Python 3.10+; polars, rapidfuzz, lightgbm, anyascii, numpy, psutil
```

Hardware used: 16 CPU cores, 64 GB RAM. Peak memory is ~45 GB (full 10M-record pool index).

## Reproduce end-to-end

Run from `src/`. Paths default to the challenge layout (`../../student_resource/dataset`,
artifacts in `../../artifacts`); override with `--data-dir` / `--work-dir`.

```bash
cd src

# 1. (optional) blocking recall report on a validation sample — train split only
python blocking.py --config ../configs/dev.json

# 2. train: blocking -> features -> LightGBM -> threshold; writes artifacts/models/dev/{model.txt,bundle.json}
python train.py --config ../configs/dev.json

# 3. test inference: writes output/matching_results.tsv + output/candidate_pairs.tsv, then runs
#    student_resource/utils/validate_submission.py
python inference.py --model-dir ../../../artifacts/models/dev

# re-decide from saved scores without re-scoring (e.g. a different threshold)
python inference.py --model-dir ../../../artifacts/models/dev --scores ../../../output/pair_scores_test.parquet --threshold 0.85
```

Diagnostics: `python evaluate.py --run <run dir>` (FP/FN/singleton error dumps, slice metrics),
`python decide_eval.py --run <run dir>` (decision-rule comparison).

## Layout

| file | role |
|---|---|
| `config.py` | dataclass config + JSON overlay + CLI / SageMaker env path resolution |
| `data_io.py` | TSV readers (UTF-8, quoting off), parquet cache |
| `normalize.py` | deterministic normalization, representations, legal-form class, skeleton |
| `dataset.py` | stable split by S1 entity |
| `blocking.py` | key generation, capped IDF index, 3-channel top-k, recall report |
| `features.py` | pair features |
| `train.py` | training entry point |
| `inference.py` | test inference entry point (+ official validator) |
| `decision.py` | exclusivity, threshold sweep, expected-F decoder (evaluated, not used) |
| `metrics.py` | official macro F0.5 (per S1, singletons included) |
| `submission.py` | TSV writers (LF line endings, exact headers) + validator call |
| `evaluate.py`, `decide_eval.py` | error analysis |

All randomness is seeded (split hash, LightGBM `seed` + `deterministic`). Every run writes its
config and metrics to `artifacts/runs/<timestamp>_<name>_*` and a row to `artifacts/experiments.tsv`.

## SageMaker

Same scripts, paths from the standard env vars: `SM_CHANNEL_DATA` (dataset root),
`SM_MODEL_DIR` (bundle), `SM_OUTPUT_DATA_DIR` (outputs / work dir). Use a CPU instance with
≥ 64 GB RAM (e.g. `ml.r5.4xlarge`); entry points `train.py` and `inference.py`, `source_dir=src`.
