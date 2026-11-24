# ER_REPRODUCIBILITY — setup, commands, artifacts, limitations

Labels: **Verified** (run in this project, and the command appears in a log), **Checked** (the argument exists in the code,
but the exact command line was not run in this form), **Untested**.
Experiment IDs refer to `ER_EXPERIMENT_REGISTRY.csv`.

---

## 1. Repository layout

`D:\ml-challenge` is **not a git repository**. There are no commits, branches or history. Code versions exist only as:

- the live tree under `code/business_entity_resolution/src/`;
- `artifacts/baseline_best/code_snapshot.zip`, the code at the time of the M7 freeze (26-09 ~06:20);
- `D:\ml-challenge_project_20260926.zip`, 1.02 GB, 156 files. It contains the code and the final model and excludes the dataset, caches and outputs.

```
D:\ml-challenge
├─ student_resource/                   official material (dataset + validator) — read-only
│  ├─ dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
│  ├─ dataset/test/test_source{1,2,3}.tsv
│  ├─ utils/validate_submission.py     OFFICIAL validator (unmodified)
│  ├─ README.md, Documentation_template.md (official blank)
├─ code/business_entity_resolution/    submission code folder (zip layout)
│  ├─ src/        13 modules, 1,590 lines (see ER_ARCHITECTURE.md §5)
│  ├─ configs/    14 JSON configs (final.json = current)
│  ├─ aws/        launch.py, sm_entry.py, validate_submission.py (copy) — SageMaker, never ran
│  ├─ README.md   OUTDATED (dev-era commands)
│  └─ requirements.txt
├─ artifacts/
│  ├─ models/<name>/  model.txt (stage 1), model2.txt (stage 2), bundle.json   — dev, dev_s2, tf30_s2, rev_tf30_s2, dryrun, final; rev63_hrt empty
│  ├─ baseline_best/  frozen M7 (models, bundle, config, code_snapshot.zip, BASELINE.json with md5s)
│  ├─ runs/<ts>_<name>_<stage>/  per-run bundle.json, threshold_sweep.tsv, val_pairs.parquet, blocking reports
│  ├─ cache/        norm_n{2,3,4}_{train,test}_s{1,2,3}.parquet, poolkeys_<hash>_{split}.parquet (3 hashes),
│  │               reverse_98cab64651_{train,test,train_dmv20}.parquet
│  └─ experiments.tsv  machine log (12 rows)
├─ lab/           research harness + experiment scripts + results.jsonl (77 rows) + dmv/ (DMV pairs & eval)
├─ output/        upload 1 files (M2)          output_rev/  upload 2 files (M7)
├─ output_s2/, output_s2_t088/   M5 files (never uploaded)
├─ submissions/   sub01_v3_t080 (NOTES.txt, bundle.json, matching_results.tsv), sub02_v4s2_t075, sub03_v4s2_t088
├─ *.log          root run logs (see ER_MASTER_HANDOVER.md §1)
├─ EXPERIMENTS.md, ROADMAP.md, Documentation_template.md (filled draft, A3-era)
└─ handover/      these six documents
```

## 2. Environment (Documented: machine used for every run)

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| CPU / RAM | 16 logical cores, 64 GB RAM; no CUDA GPU (Intel Iris Xe) |
| Python | 3.13 (miniconda at `C:\Users\dhari\miniconda3`) |
| Packages | `pip install -r code/business_entity_resolution/requirements.txt`: polars 1.42.1, numpy 2.3.5, rapidfuzz 3.14.6, lightgbm 4.7.0, anyascii 0.3.3, psutil 7.2.2 (Verified with `pip show`) |
| Shells | Git Bash or PowerShell. On Windows set `PYTHONIOENCODING=utf-8` for ad-hoc scripts; the entry points call `sys.stdout.reconfigure(encoding="utf-8")` themselves |
| Power | Keep the laptop plugged in: on battery, chunks run about 3× slower (I4) |

**Memory.** Final training peaked at **44.21 GB** RSS. A first attempt without the `del data` / `del tr` fix ran low on free
RAM and paged (M11a). Machines with less than about 48 GB RAM need a lower `train_frac` or `es_frac`, which changes the model.

## 3. Dataset preparation

No preparation is needed. `config.Config.data_dir` defaults to `../../student_resource/dataset`, relative to
`code/business_entity_resolution/`, and expects the official `train/` and `test/` folders unchanged. The files are read with
`quote_char=None` and all columns as strings; missing values become "".

**Record counts** (Verified from the n4 caches):

| Split | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

## 4. Configuration

- JSON files overlay the dataclass defaults in `config.py`. Unknown keys raise `KeyError`. `--config` paths are resolved
  against the current working directory, so run from `src/` with `../configs/<name>.json`.
- CLI flags common to all entry points: `--config`, `--data-dir`, `--work-dir`.
- Environment variables read: `SM_CHANNEL_DATA`, which sets the data dir; `SM_OUTPUT_DATA_DIR`, which sets the work dir and
  the inference output dir; `SM_MODEL_DIR`, which sets the model dir. These exist for SageMaker. None is needed locally.
- **Current config:** `configs/final.json` (full text in ER_ARCHITECTURE.md §8).
- Other configs are history: dev, dev_cap1000/2000, dev_s2, tf30_s2, rev_s2, rev_tf30_s2, rev63, rev63_house, rev63_hrt,
  rev63_hrt_tb (the hash tie-break variant, never run), smoke (SageMaker smoke test, never ran) and dryrun.

## 5. Commands

All commands run from `D:\ml-challenge\code\business_entity_resolution\src`.

| Purpose | Command | Status | Time and memory (this laptop) | Writes |
|---|---|---|---|---|
| Normalizer self-test | `python normalize.py` | Checked (asserts in `__main__`) | seconds | nothing |
| Metric self-test | `python metrics.py` | Checked | seconds | nothing |
| Blocking recall report | `python blocking.py --config ../configs/final.json [--split train]` | Verified for earlier configs (B1–B5 logs) | 2–10 min with warm caches; ~70 min to build the reverse cache if it is missing | `artifacts/runs/<ts>_final_blocking_train/`, experiments.tsv row |
| **Train the final model** | `python train.py --config ../configs/final.json` | Verified (`train_final.log`, EXIT 0) | 2,980 s, 44.2 GB peak, with warm caches | ⚠ **overwrites `artifacts/models/final/`**. Pass `--model-dir <new dir>` to keep the current checkpoint |
| Dry run of training | `python train.py --config ../configs/dryrun.json` | Verified (V11) | 68 s | `artifacts/models/dryrun/` |
| Dry run of inference | `python inference.py --model-dir ../../../artifacts/models/dryrun --max-s1 5000 --out-dir <dir>` | Verified (V11) | ~1 min | the out dir; validator skipped |
| **Test inference, final model** | `python inference.py --model-dir ../../../artifacts/models/final --out-dir ../../../output_final` | **Untested with this model; NOT approved by the owner** (P3). The same command shape ran for M7 (`inf_rev.log`) | M7 took 12,154 s (4,453 s prepare including the test reverse-cache build, then ~70–110 s per 100k S1). The reverse test cache now exists, so expect about 2.5 h (Inferred) | `matching_results.tsv`, `candidate_pairs.tsv`, `pair_scores_test.parquet` (~2 GB), then runs the official validator |
| Re-threshold saved scores | `python inference.py --model-dir <M> --scores <out>/pair_scores_test.parquet --threshold 0.85 --out-dir <new dir>` | Checked (arguments exist); untested end to end | ~2–5 min (Documented estimate) | new output files |
| Error dump | `python evaluate.py --config <cfg> --run <run dir> [--t 0.8] [--n 400]` | Checked (arguments exist) | minutes | files in the run dir |
| Decision-rule comparison | `python decide_eval.py --config <cfg> --run <run dir>` | Checked | minutes | stdout |
| Official validation | `python ../../../student_resource/utils/validate_submission.py` (see the file header for arguments such as `--candidate` and `--check-ids`) | Verified: `inference.py` calls it and printed "validator PASS" for output_rev | minutes | stdout |
| Trace test | `python ../../../lab/trace_test.py` | Verified 8/8 (V10) | minutes | stdout |
| Lab harness | `python ../../../lab/exp_*.py`, `lab/dmv_harness.py` | Verified (logs in `lab/`) | 5–60 min each | `lab/results.jsonl` (append) |

⚠ **Output-folder trap.** Without `--out-dir`, `inference.py` writes to `<work_dir>/../output`, which is `D:\ml-challenge\output`.
That folder holds the **upload-1** files. Always pass `--out-dir`.

⚠ **Model-folder trap.** `train.py` writes to `artifacts/models/<config name>` unless `--model-dir` is given. Re-running a
config overwrites its checkpoint. This is how the M1 files were lost when M2 reused the name `dev`.

## 6. Submission packaging (not yet done)

The required zip is `<team_name>_submission.zip`, containing:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `code/business_entity_resolution/{src/, README.md, requirements.txt}`
- `Documentation_template.md`

Blocked on the following:

- **The final test output does not exist yet.** It will be created by P3.
- **The team name and members are unknown.**
- **`README.md` and `Documentation_template.md` describe older models** (A1 and A3), not the final one.

## 7. Input and output formats

- **Input TSV per source:** `entity_id`, `business_name`, `business_address`, `country`, read as strings. Train and test
  S1 have no empty addresses; about 2.7–3.4% of S2/S3 addresses are empty (Verified: n4 caches).
- **Ground truth:** `train_ground_truth.tsv` with columns `source1_entity_id` and `matched_entity_ids` (comma-separated,
  empty for singletons).
- **`matching_results.tsv`:** header `source1_entity_id<TAB>matched_entity_ids`. One row per test S1 in file order; the IDs
  are comma-joined and ordered by score; an empty field means no match; LF line endings.
- **`candidate_pairs.tsv`:** header `source1_entity_id<TAB>candidate_entity_ids`. This must be exactly the blocking candidate
  set, and matches must be a subset of it. The official validator warns, but does not fail, otherwise.

## 8. Checkpoints (no pretrained weights anywhere)

| Model | Files | Val macro F0.5 | Notes |
|---|---|---|---|
| **final (M11)** | `artifacts/models/final/{model.txt, model2.txt, bundle.json}` | 0.9739 @ t=0.80 | current. bundle.json holds the resolved config, features, threshold and metrics |
| rev_tf30_s2 (M7) | `artifacts/models/rev_tf30_s2/` and the frozen copy `artifacts/baseline_best/` | 0.9691 (0.9688 exclusive) | produced upload 2. md5: model a973f574…, model2 12244752… |
| dev (M2) | `artifacts/models/dev/` and a copy of the bundle in `submissions/sub01_v3_t080/bundle.json` | 0.9584 | produced upload 1 |
| dev_s2, tf30_s2, dryrun | `artifacts/models/<name>/` | 0.9598, 0.9615, 0.9653 (1% subset) | history |

`inference.py` rebuilds the pipeline configuration from `bundle.json`. Any model folder is therefore self-describing,
**provided the code has not changed its feature definitions since the model was trained** (see §9).

## 9. Reproducibility limitations and technical debt

1. **No version control.** Code changed between models: A0–A3 were trained with earlier code. Only
   `baseline_best/code_snapshot.zip` preserves an older state, so older bundles may not be re-runnable bit-exactly with
   the current code. For example, the 0/0 → null fix to `score_*_rel` was made after M7 was trained. **Inferred risk:**
   M7 inference with the current code sees null where it saw NaN. LightGBM treats both as missing, so the impact is
   likely nil, but this was not measured.
2. **Caches are keyed by version tag, not by content.**
   - `norm_n4_*` is keyed only by `normalize.VERSION`. Editing `normalize.py` without bumping VERSION silently reuses stale caches.
   - `poolkeys_<hash>` and `reverse_<hash>` are keyed by `stage_hash(config section)` plus the polars version, but **not by code**.
   - Delete `artifacts/cache/` to force a rebuild. A cold build takes about 2.5 h more per split for the reverse cache, plus pool keys and normalization.
3. **Determinism.**
   - LightGBM: `deterministic=True`, `seed=42`, `force_row_wise=True`.
   - Splits: hash-based and seeded.
   - Blocking: ties are ordered by `rid`.
   - V11 showed byte-identical inference outputs on a repeat run.
   - Full-scale training has not been repeated, so its run-to-run variance is Unknown.
   - Lab seed noise is ±0.0002–0.0004.
4. **Rank tie bias.** `tie_break="rid"` (B9) is kept on purpose for compatibility with the caches and the trained models. `"hash"` is implemented but unvalidated (B10).
5. **`kt_NS` is a constant-false feature.** The NS key is not in `key_types`, yet the feature is still in the list. It is harmless but dead.
6. **Exclusive-decoder ties.** Rows with exactly equal max p for the same record are all kept, so exclusivity is not strict. The frequency of this is Unknown.
7. **Threshold choice.** `best_threshold` takes the strictest t within 0.001 of the maximum. For M11 the maximum is 0.97472 at t=0.675, but the chosen t is 0.80 (0.97390). This is a deliberate hedge toward precision for the denser test pool (V5), and it is not validated on test.
8. **Reported val metrics are optimistic by about 0.001**, from threshold selection on the same fold (V6). The lab research (R1–R14, V3–V7) also reused **fold 0** (val) pairs for the feature decisions, so M11's 0.9739 is not a clean held-out estimate. The size of this selection bias is Unknown.
9. **Stale documents.**
   - `code/business_entity_resolution/README.md` describes the dev-era pipeline.
   - The root `Documentation_template.md` describes A3.
   - `EXPERIMENTS.md` and `ROADMAP.md` stop before M11 (last updated around 08:30 on 26-09).
10. **The SageMaker path (`aws/`) has never completed a job** (quota 0). It is untested.
