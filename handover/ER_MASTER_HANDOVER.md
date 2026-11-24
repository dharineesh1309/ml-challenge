# ER_MASTER_HANDOVER: Business Entity Resolution (Amazon ML Challenge 2026)

_Audit date: 26 Sep 2026, ~13:30–14:30 IST. This is a read-only audit: no code, config, model or checkpoint was changed.
Submission deadline: **27 Sep 2026, 23:59 IST**._

**Document map**

| File | Content |
|---|---|
| `ER_MASTER_HANDOVER.md` (this file) | The source of truth. The executive summary is at the end (§12) |
| `ER_EXPERIMENT_REGISTRY.csv` | 70 experiments, one row each, with evidence paths. The IDs are used everywhere |
| `ER_ARCHITECTURE.md` | Architecture versions A0–A4, three Mermaid diagrams, component reference, final config |
| `ER_RESEARCH_BIBLIOGRAPHY.md` | Papers, repositories, blogs and libraries, with implementation status |
| `ER_REPRODUCIBILITY.md` | Setup, verified commands, checkpoints, reproducibility limits, technical debt |
| `ER_KNOWLEDGE_GAPS.md` | Missing information (G), contradictions (U), assumptions (A), owner questions (Q) |

**Confidence labels**

| Label | Meaning |
|---|---|
| **Verified** | I read or measured it during this audit: code, bundles, logs, caches, sweep tables, ground truth |
| **Documented** | Recorded in project notes (EXPERIMENTS.md, ROADMAP.md) or the session transcript, but not re-measured |
| **Inferred** | My reasoning |
| **Proposed** | An idea that was never implemented |
| **Unknown** | Not determinable |

---

## 1. Evidence inventory

| Evidence | Location | Inspected | Notes |
|---|---|---|---|
| Source code (13 modules, 1,590 lines) | `code/business_entity_resolution/src/` | Verified: all modules | No git. Older code state only in `artifacts/baseline_best/code_snapshot.zip` (M7 era) |
| Configs (14) | `code/business_entity_resolution/configs/` | Verified | `final.json` is current |
| SageMaker launcher | `code/business_entity_resolution/aws/` | Verified (exists) | Never completed a job |
| Model bundles (6 + empty rev63_hrt) | `artifacts/models/*/bundle.json` | Verified: every metric extracted | M1's bundle was overwritten by M2 (same name `dev`) |
| Frozen baseline | `artifacts/baseline_best/` | Verified | M7 plus md5s |
| Run folders (15) | `artifacts/runs/` | Verified (listing, the final sweep table) | `20260925_192640` has no report (crash) |
| Machine experiment log | `artifacts/experiments.tsv` (12 rows) | Verified | Blocking and training rows only |
| Human experiment log | `EXPERIMENTS.md` (§A blocking, §B models, §R research, §V validation, §C submissions, §D infrastructure) | Verified (read) | Stops before M11. Documented-level evidence |
| Roadmap | `ROADMAP.md` | Verified (read) | Header says 26-09 03:15; contains a "STATUS 26-09 08:30" block; nothing later |
| Lab harness + results | `lab/harness.py`, `lab/exp_*.py`, `lab/results.jsonl` (77 rows), `lab/dmv/baseline_eval.json`, `lab/*.log` | Verified | Research experiments R1–R14, V1–V9 |
| Root logs | `*.log` (m1_run*, cap*, m2_train*, train_*, inf_*) | Verified (headers and tails) | |
| Outputs | `output/`, `output_rev/`, `output_s2/`, `output_s2_t088/`, `submissions/` | Verified (listing, sizes, times) | |
| Caches | `artifacts/cache/` | Verified (listing; record counts and missingness read from the n4 caches) | |
| Dataset | `student_resource/dataset/` | Verified: ground-truth statistics computed. Source TSVs **not** fully scanned (per the owner's instruction), so statistics come from the caches | |
| Official validator | `student_resource/utils/validate_submission.py` | Verified (headers); unmodified | |
| Session transcript | `C:\Users\dhari\.claude\projects\d--ml-challenge\7599aa93-….jsonl` | Verified: targeted extraction (URLs, research report, final analysis, handover prompt) | The only record of pasted prompts and leaderboard reports |
| Problem-statement PDF | `D:\6ab10eb3b23ba_student_resource\…problem_statement.pdf` | **Not re-read in this audit** (read at project start) | ER_KNOWLEDGE_GAPS G12 |
| Leaderboard, AWS, A100, Kaggle | external | **Not accessible** | Leaderboard numbers come from the owner |
| External papers and repositories | web | **Not re-fetched** | Taken from the project's earlier research write-ups |
| Competitor challenge repositories | web | **Deliberately not opened** | Plagiarism clause |

---

## 2. Problem and objective

### 2.1 Task requirements (from the project notes, the official README and validator, and the start-of-project reading of the problem statement: Documented)

- **Entity resolution in this project.** Link every Source-1 business record (clean and deduplicated) to **all** records in
  Source 2 and Source 3 that describe the same real-world business. An S1 can have 0 to n matches.
- **Output.** For each test S1, a set of S2/S3 IDs. It is not a clustering, and S2↔S2 links are not required.
- **Sources.** Three independent business-data sources. Every source has the same four columns: `entity_id`,
  `business_name`, `business_address`, `country`. IDs are prefixed `S2-` / `S3-`.
  - **S1** is clean: Latin script, no empty addresses (Verified).
  - **S2/S3** are noisy: typos, OCR-like digit/letter swaps, native Indic script, legal-form variants, abbreviations,
    reordered or missing address parts, some empty addresses, house-number noise (Verified from the caches and error analysis).
- **Metric.** Macro F0.5 over S1 entities.
  - F0.5 = 1.25·P·R / (0.25·P + R) per S1.
  - An S1 with no true and no predicted matches scores 1. A predicted set for a true singleton scores 0. No predictions for a non-singleton scores 0.
  - The metric is precision-weighted. Implementation: `src/metrics.py`, which reproduces the statement's example of 0.714.
- **Submission files.**
  - `output/matching_results.tsv` with header `source1_entity_id<TAB>matched_entity_ids`. This file is scored.
  - `output/candidate_pairs.tsv` with header `source1_entity_id<TAB>candidate_entity_ids`. Matches must be a subset of it.
  - Zip `<team>_submission.zip` = `output/` + `code/business_entity_resolution/{src,README.md,requirements.txt}` + `Documentation_template.md`.
  - Check with the official `utils/validate_submission.py`.
- **Constraints.**
  - Models must be MIT or Apache licensed and ≤ 8B parameters.
  - No external data, APIs, geocoding or internet augmentation.
  - 5 leaderboard uploads per day.
  - The public leaderboard is a subset of test.
- **Leaderboard context (owner-reported).** Rank 1 is about 0.981. Our best is 0.95099 (rank 603 after upload 1).

### 2.2 Owner and working assumptions (not task requirements)

- **CPU-first.** One config-driven pipeline for the laptop, Kaggle and SageMaker (owner's design prompt). No GPU was available locally.
- **France is open-set.** France appears only in test (Verified). No country-specific model is trained, and there is no country feature.
- **Exclusivity.** Each S2/S3 record belongs to at most one S1. This is true in train ground truth: 0 shared records (Verified) (A1).
- **Test density.** The test pool has about 40% unmatched records versus 26% in train (Inferred, A2), which motivated DMV (V1).
- **Research hygiene.** Competitor repositories are not inspected. Measure rather than fabricate.
- **Approvals.** Long jobs, packaging and uploads need owner approval (owner's instructions; see memory `ask-before-long-jobs`).

---

## 3. Chronology and experiment registry

### 3.1 Timeline (IST)

| When | Phase | IDs |
|---|---|---|
| 25-09 afternoon | Read problem statement and data; strategy research (Foursquare, Sparkly, Splink, …) | — |
| 25-09 19:26–21:09 | Blocking design | B0–B5, B8, N1, N2 |
| 25-09 20:29–23:07 | First models, error analysis, decision rules, stage 2 | M1–M5, N3, I7 |
| 25-09 22:55 | **Leaderboard upload 1 (M2): 0.95099** | S1 |
| 26-09 00:05–00:41 | M5 test outputs (never uploaded); 3× data | S0, M6 |
| 26-09 00:42–02:29 | Reverse blocking and M7 (best at the time) | B6, M7, L2 |
| 26-09 02:30–05:51 | M7 test inference (3.4 h) | S2 |
| 26-09 03:00–04:41 | Research pass in the lab harness | B7, R1–R17, B9 |
| 26-09 ~05:55 | **Leaderboard upload 2 (M7): "same 0.951"** | S2 |
| 26-09 05:52–06:12 | rev63 failed to start; rev63_hrt stopped at round ~800 | M8, M9 |
| 26-09 06:20–08:19 | Architecture validation: DMV, ablations, France, trace test, dry run | V1–V11, B10 |
| 26-09 ~09:00 | Kaggle packaging interrupted; SageMaker quota 0 | I5, I2, I3 |
| 26-09 ~11:30 | Project zip | I6 |
| 26-09 12:16–13:19 | Final training (first attempt aborted for memory) | M11a, M11, L3 |
| 26-09 13:20 → | This audit | — |

### 3.2 Experiment table

Full fields (dataset, config, runtime, influence, confidence) are in `ER_EXPERIMENT_REGISTRY.csv`. Validation =
**official macro F0.5 on 44,736 fold-0 S1 against the full 10.32M train pool**, unless marked *lab*.

> ⚠ *Lab* numbers use half the data and stage 1 only (R1–R8), or a stage-2 lab variant (R9–R14, V3–V7). Their absolute
> values are **not comparable** with bundle numbers. Compare deltas within the same control only.

| ID | Experiment | Hypothesis / change | Baseline | Actual results | Outcome | Evidence |
|---|---|---|---|---|---|---|
| B0 | First blocking run | first index implementation | — | crashed, no report | abandoned | runs/20260925_192640 |
| B1 | 4 key types, top-50 | NT, AW, HS, CAT; cap 5000 | — | recall 0.9293, 99.3 cand/S1 | baseline | experiments.tsv r1 |
| N1 | Norm n2 | GENERIC-word stripping | n1 | Not found separately | kept | EXPERIMENTS B2 |
| B2 | Channels + NP + NS | name/addr/comb top-k | B1 | 0.9328, 52.1 cand/S1 | kept | experiments.tsv r2 |
| B3 | NX cross key, −NS | skeleton × address word | B2 | 0.9490, 48.3 cand/S1 | kept | experiments.tsv r3 |
| M1 | dev v1 LightGBM | 49 features, lr 0.05 / 127 leaves | — | 0.9406 @0.75, P 0.986 R 0.880 | baseline | experiments.tsv r4 |
| I7 | First test inference | cap 5000 | M1 | 855 s / 100k S1 → killed | abandoned | inf_dev1.log |
| N2 | Norm n3 | ordinals, state codes | n2 | Not found separately | kept | EXPERIMENTS B4 |
| B4 | n3 + cap 2000 | | B3 | 0.9530, 51.5 cand/S1 | kept | experiments.tsv r5 |
| B5 | cap 1000 | faster | B4 | 0.9517, 50.5 cand/S1, ~3.5× faster | **chosen** | experiments.tsv r6 |
| B8 | k-budget grid | Pareto | B3–B5 | chose 20/10/10 | kept | EXPERIMENTS §A |
| M2 | dev v3 | legal form, house prefix/edit, lr 0.1 / 63 leaves, 54 features | M1 | **0.9584** @0.80, P 0.9916 R 0.9053 | kept | models/dev/bundle.json |
| S1 | **Upload 1** | M2 @0.80 | — | **LB 0.95099**, rank 603 | uploaded | submissions/sub01_v3_t080 |
| M3 | Expected-F0.5 decode | per-S1 optimal set | M1 0.9406 | 0.9389 (singletons 0.790) | rejected | EXPERIMENTS M3 |
| M4 | Exclusivity | record → best S1 | M1 | 0.9406 (neutral) | kept | EXPERIMENTS M4 |
| N3 | Norm n4 | OCR fix, lnc, and-co | M2 | 0.9591 alone | kept | EXPERIMENTS M5 |
| M5 | dev_s2 | n4 + stage 2 | M2 | 0.9598 @0.75, singleton 0.9744 | kept | models/dev_s2 |
| L1 | Loss split M5 | | | blocking 0.0171 / classifier 0.0156 / false merge 0.0076 | → B6 | EXPERIMENTS |
| S0 | sub02/sub03 | M5 @0.75/0.88 | | files only | not uploaded | submissions/ |
| M6 | tf30_s2 | train_frac 0.3 (20.0M pairs) | M5 | 0.9615 @0.775 | kept | models/tf30_s2 |
| B6 | Reverse blocking | pool→S1 index, add top-5 | B5 | recall **0.9706**, 66.1 cand/S1 | kept | baseline_eval.json |
| M7 | rev_tf30_s2 | + reverse candidates and ranks (60 features) | M6 | **0.9691** @0.80, P 0.9938 R 0.9312, oracle 0.9895 | kept, frozen | models/rev_tf30_s2; baseline_best |
| L2 | Loss split M7 | | | 0.0105 / 0.0153 / 0.0050 | → research | EXPERIMENTS |
| S2 | **Upload 2** | M7 @0.80 | S1 | LB "same 0.951" (exact Not found) | uploaded | output_rev; inf_rev.log |
| B7 | Char n-gram TF-IDF | typo misses | B6 | analysis only (est. +0.001) | parked | EXPERIMENTS B7 |
| R1 | House change type | digit diff/position/transposition | *lab* 0.9625 | 0.9649 / 0.9646 | **kept** (+0.0023) | results.jsonl house |
| R2 | Missing-address interactions (Ong) | | *lab* 0.9625 | 0.9623 | rejected | results.jsonl batch2 |
| R3 | Empty-address specialist | | *lab* 0.9625 | 0.9625 | rejected | batch2 |
| R4 | Reverse margins (Selective EM) | competition features | *lab* 0.9625 | 0.9640 / 0.9638 | **kept** (+0.0014) | batch3/4 |
| R5 | S1-side commonness + n_channels | | *lab* 0.9625 | 0.9628 | inconclusive → dropped | results.jsonl |
| R6 | Hard-negative ×3 | | *lab* 0.9625 | 0.9624 | rejected | batch4 |
| R7 | LambdaRank by S1 | listwise | *lab* 0.9625 | 0.9575 | rejected (−0.005) | batch4 |
| R8 | R1+R4 | additivity | *lab* 0.9625 | 0.9664 / 0.9664 | kept | batch5 |
| R9 | One-hop profile (TransClean) | seeds p1 ≥ 0.9, 8 features | *lab-s2* 0.9688 | 0.9703 / 0.9700 | **kept** (+0.0013) | batch5 |
| R10 | Stage 2 + R1 + R4 + R9 | | *lab-s2* 0.9688 | 0.9718 / 0.9716 | **kept** (+0.0029) | batch5 |
| R11 | Cross-country transfer | France proxy | — | US→US 0.970, **US→IN 0.847**, IN→IN 0.953, IN→US 0.947 | warning | xcountry |
| R12 | Feature removal for transfer | | US→IN 0.847 | best 0.864; in-country −0.003 to −0.005 | rejected | robust |
| R13 | Cross-source corroboration (GraLMatch) | | *lab-s2* 0.9688 | 0.9692 / 0.9686 | rejected | xsrc |
| R14 | Strict profile p1 ≥ 0.95 | | *lab-s2* 0.9688 | 0.9696 / 0.9698 | not adopted | xsrc |
| R15 | ING EMM features | | — | not run | already covered | EXPERIMENTS §R |
| R16 | Multilingual / IndicXlit | | — | not run | rejected (4% of misses) | EXPERIMENTS §R |
| R17 | Source-aware model | | — | not run | parked | EXPERIMENTS §R |
| B9 | Tie-break bias | rid ties favour early S1 | B6 | adds/S1 97→20 across the file; recall 0.977 vs 0.967 | bug confirmed | EXPERIMENTS §R |
| B10 | Hash tie-break | `tie_break=hash` | B9 | not run | implemented, unvalidated | blocking.py |
| M8 | rev63 | +s1_cat_df, n_channels | M7 | never started (wrong working directory) | failed | train_rev63.log |
| M9 | rev63_hrt | rev63+R1+R4+R9 | M7 | stopped at round ~800 | interrupted | train_rev63_house.log |
| V1 | DMV | test-like density | M7 | 0.9688 → 0.9683 | density ≠ main gap | baseline_eval.json |
| V2 | Stage 1 vs 2 | | M7 | 0.9687 vs 0.9688 | stage 2 needs data | baseline_eval.json |
| V3 | Ablations @DMV | | *lab* 0.9685 | full 0.9713/0.9714; full−rev63 0.9714/0.9717 | keep R1/R4/R9, drop rev63 | dmv-M1 |
| V4 | Ablations @normal | | *lab* 0.9686/0.9687 | full 0.9717/0.9721 | consistent | dmv-M2 |
| V5 | DMV training | | | 0.9711 vs 0.9713/0.9714 | not needed | dmv-M3 |
| V6 | Threshold-selection bias | | | cost ≈ 0.001 | val ≈ 0.001 optimistic | dmv-M3/M4 |
| V7 | Stage-2 learning curve | | | 0.9701 / 0.9707 / 0.9713 | es_frac 0.5 | dmv-M4 |
| V8 | France street conflict | label-free | | FR 3.5% / US 1.5% / IN 1.3% | France mode found | EXPERIMENTS V8 |
| V9 | IDF address features | fix for V8 | | ±0.0001; transfer −0.0009 | rejected | idf |
| V10 | Trace universe | 13 pairs, 8 cases | | 8/8 pass; NaN bug fixed | passed | lab/trace_test.py |
| V11 / M10 | Dry run (train + inference ×2) | | | 68 s; byte-identical; validator 0 errors; 0.9653 on 1% (not comparable) | passed | lab/dryrun_* |
| I6 | Project zip | | | 1.02 GB, 156 files | done | D:\ml-challenge_project_20260926.zip |
| M11a | Final, attempt 1 | | | aborted (RAM) | aborted | runs/20260926_121631 |
| **M11** | **final** | M7 + R1 + R4 + R9; stage-1 ES on train slice; es_frac 0.5; drop rev63 features | M7 0.9691 | **0.9739 @0.80**, P 0.9962 R 0.9401, FP 561 FN 9,264, singleton 0.9864, oracle 0.9895 | **current** | models/final; train_final.log |
| L3 | Loss split M11 | | L2 | 0.0105 / 0.0123 / 0.0033 | | val_pairs.parquet |
| I1–I5 | Leak check; SageMaker ×2; battery; Kaggle | | | no leak; quota 0; 3× slower; interrupted | | EXPERIMENTS §D |
| P1–P4 | France probe; GPU arch; final test inference; LB threshold calibration | | | not run | proposed / planned | see §9, §11 |

### 3.3 Detailed notes on the experiments that changed the architecture

**Blocking, B1 → B6 (recall 0.929 → 0.971).**
- *Starting point.* The first design (B1) used one top-50 list per source. Recall was 0.9293, i.e. 7.1% of true pairs
  lost. EXPERIMENTS.md says "5% of pairs lost", which contradicts its own recall figure (ER_KNOWLEDGE_GAPS U7). The true
  match was ranked first only 43% of the time.
- *B2: channels.* Splitting retrieval into name / address / combined channels kept recall with half the candidates. The
  skeleton unigram key (NS) recovered nothing unique.
- *B3: cross key.* The skeleton × address-word key (NX) targets transliterated names that share a street or city.
  Recall rose to 0.949, and native-script recall went from 0.71 to 0.85 (Documented).
- *B4–B5: normalization and cap.* norm n3 plus a lower df cap raised recall. Cap 1000 was chosen for about 3.5× faster
  queries at a cost of −0.001 recall.
- *B6: reverse blocking, the largest single gain.*
  - Error analysis L1 showed that blocking misses were the largest loss (0.0171).
  - A probe (Documented) found that 97.9% of true pairs have the S1 as the pool record's reverse top-1.
  - So every pool record queries an S1 index, and its top-5 S1s are added as candidates. This recovered 37% of misses:
    recall 0.9706, 66.1 candidates per S1.
  - The same reverse ranks became the strongest features. `rrank_c` was the top M7 feature at about 8× the next.
- *B9: a defect in B6.* Reverse-rank ties are broken by S1 row order, so S1s early in the file win more reverse adds
  (97 → 20 per S1 across the test file). A fix exists but is unvalidated (B10).

**Classifier, M1 → M11 (0.9406 → 0.9739).**
- *M2 (+1.78 points).* Driven by error analysis: legal-form class and conflict, house-number prefix and edit distance,
  ordinal and state normalization. It also switched to lr 0.1 and 63 leaves for inference speed.
- *M5 (+0.0014).* Added a stage-2 model over per-S1 context of stage-1 scores. It mainly helped singletons (0.966 → 0.974).
- *M6 (+0.0017).* Three times the training data.
- *M7 (+0.0076).* Reverse candidates and reverse ranks.
- *Research pass.* The M7 loss split (L2) showed classifier misses as the largest loss (0.0153). 36% of those misses have
  an empty candidate address, and house-number mismatch accounts for 37% of misses.
- *Features kept from the lab harness.*
  - R1: house-change type.
  - R4: reverse-competition margins. Empty-address false merges fell from 170 to 70–85.
  - R9: one-hop S1 profile in stage 2. House-difference misses fell by 24%.
- *Additivity.* The three combine to +0.0029 (R10), and the gain holds at test-like density (V3, V4).
- *Rejected on measurement.* LambdaRank (R7: −0.005, because it loses cross-S1 calibration), hard-negative weighting (R6),
  missingness interactions (R2, R3), cross-source-only corroboration (R13), and feature removal for transfer (R12).
- *M11: validation-phase fixes on top.* M11 adds the R1, R4 and R9 features, plus:
  - Stage 1 now early-stops on a training slice, so stage-2 inputs are fully out-of-fold.
  - Stage 2 gets five times more data (`es_frac` 0.5, V7).
  - The rev63 features, which showed no value, are dropped (V3).
  - The threshold sweep now uses the inference decoder.
- *M11 result.* 0.9739 at t=0.80 (+0.0048 over M7 at the same threshold); precision 0.9962.

**Decision.**
- The global threshold beat per-S1 expected-F decoding (M3).
- Exclusivity is kept (M4). It is neutral on validation and principled: train ground truth has 0 shared records.
- `best_threshold` picks the **strictest** t within 0.001 of the best. For M11 the sweep maximum is 0.97472 at t=0.675,
  but the chosen t is 0.80 (U2). This is a hedge toward precision for the denser test pool (V5) and is not validated on test.

**Validation-to-leaderboard gap.**
- Upload 1 had a gap of 0.0074. Upload 2 was reported as unchanged at about 0.951, despite +0.0107 on validation (U1).
- Density explains only 0.0005 (V1).
- France is 15% of test S1 and absent from train. Transfer drops sharply between countries (R11). France also shows a
  specific street-conflict false-merge mode (V8).
- The attempted country-agnostic fix (V9) showed no measurable benefit and was not integrated.
- The leading hypothesis is France (Inferred). It is untested: P1 is not done and G4 is unknown.

---

## 4. Current architecture: A4 (M11, `configs/final.json`). Details in `ER_ARCHITECTURE.md`

```
TSV (3 sources) → data_io.read_tsv → normalize.normalize_records (n4, cached)
  → features.add_name_commonness → blocking.pool_keys / build_index (6 key types, df ≤ 1000, IDF)
  → blocking.query (forward top 20/10/10 per S1×source)  ∪  blocking.reverse_ranks (pool→S1, add best rank ≤ 5)
  → candidate_pairs.tsv (exact candidate set)
  → features.pair_features (70) → LightGBM stage 1 (p1)
  → features.s1_context (10) + features.transitive_support (8) → LightGBM stage 2 (88 → p)
  → decision.exclusive (record keeps only its argmax-p S1) → decision.apply_threshold (t = 0.80)
  → submission.write_outputs → official validator
```

- **Implemented.** Everything above. No neural components, no embeddings, no pretrained weights (Verified).
- **Not present.**
  - Graph construction, connected components and S2↔S3 clustering. Transitivity is used only as stage-2 features (R9).
  - Probability calibration: raw LightGBM probabilities plus a global threshold.
  - Country-specific logic.
- **Partially implemented or experimental.**
  - `tie_break="hash"` (B10).
  - `decision.expected_f_decode` (rejected in M3, still in the code).
  - SageMaker launcher (never ran).
  - `kt_NS` feature (constant false).

Architecture versions (details in ER_ARCHITECTURE.md §1):

| Version | Model | Val macro F0.5 | Status |
|---|---|---|---|
| A0 | M1 | 0.9406 | superseded |
| A1 | M2 | 0.9584 | upload 1 |
| A2 | M5/M6 | 0.9598 / 0.9615 | superseded |
| A3 | M7 | 0.9691 | frozen baseline, upload 2 |
| **A4** | **M11** | **0.9739** | **current, not yet run on test** |

---

## 5. Current model details (Verified from `artifacts/models/final/bundle.json`, `train.py`, `features.py`)

**Learner.** LightGBM 4.7.0 GBDT, objective `binary`. The loss is binary log-loss
L = −Σ[y·log p + (1−y)·log(1−p)], with no class weights.

**Parameters.** Learning rate 0.1, `num_leaves` 63, `min_data_in_leaf` 100, `feature_fraction` 0.8,
`bagging_fraction` 0.8 with `bagging_freq` 1, `lambda_l2` 1.0, `seed` 42, `deterministic`, `force_row_wise`.
Up to 3,000 rounds with early stopping after 100 rounds on logloss. There is no batch size (full-data GBDT).

**Stage 1.**
- 70 float32 features (full list in ER_ARCHITECTURE.md §5.6); 1,688 trees.
- Trained on 90% of the train-role S1 (396,588 S1 give 26.3M pairs, 5.07% positive).
- Early-stopped on the other 10%, selected by `(s1·2654435761 mod 1000) < 100`.
- Top features by gain: `rrank_a`, `rrank_c`, `r_gap_best`, `r_ratio`, `num_jacc`, `house_absdiff`, `r_margin12`, `name_partial`.

**Stage 2.**
- 88 features = the 70 stage-1 features + 10 `S1_CONTEXT` + 8 `TRANSITIVE`; 169 trees.
- Data is the es role: fold 1, bucket < 500, i.e. 221,123 S1 and 14.6M pairs. Even S1 ids are used for fitting and odd
  ids for early stopping.
- Top features: `p1`, `rrank_c`, `p1_gap`, `r_ratio`, `r_gap_best`, `p1_rank`.

**Negatives.** Every blocking candidate that is not a ground-truth pair. Negatives are therefore naturally hard, since they
passed IDF blocking. There is no random negative sampling and no re-weighting (R6 was rejected).

**Blocking score.** For each shared key k:
- `idf(k) = ln(N_country / df(k))`, and keys with df > 1000 are dropped;
- `score_n = Σ idf` over the name-side keys (NT, CAT, NP, NX; NS is also counted, but it is not a generated key type in
  A4, so it contributes nothing);
- `score_a = Σ idf` over the address-side keys (AW, HS, NX);
- `score = score_n + score_a`.

Ranks are computed per S1 × source. Reverse ranks are the same computation on an S1-side index queried by each pool record.

**Similarities.** rapidfuzz ratio, token_set_ratio, token_sort_ratio, partial_ratio, Jaro-Winkler and Levenshtein, plus
Jaccard and intersections over signature tokens, address words and numbers.

**Decision.**
- p* = max over S1 of p(s1, r) for each record r. A pair is kept if p = p* and p ≥ 0.80.
- There is no calibration step.

**Runtime.**
- Training: 2,980 s, peak 44.21 GB.
- Test inference: M7 took 12,154 s. M11 is expected to be similar or slightly slower because of the transitive features (Documented estimate: +5–10 min).

**Environment.** See ER_REPRODUCIBILITY.md §2.

---

## 6. Dataset, splits and leakage

### 6.1 Data (Verified)

| | S1 | S2 | S3 | Countries (S1) |
|---|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | US 1,323,633 (60.0%) · India 883,188 (40.0%) |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | India 809,986 (46.8%) · US 663,106 (38.3%) · **France 259,452 (15.0%)** |

**Train ground truth** (`train_ground_truth.tsv`, one row per S1):
- 7,638,365 true pairs: S2 3,693,619 and S3 3,944,746. About 73–75% of each pool is matched, so 26% of the pool is unmatched.
- Matches per S1: mean 3.46, maximum 11.
- **Singletons: 123,247 (5.6%).**
- Distribution of matches per S1:

  | Matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | ≥10 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | S1 count | 123,247 | 119,157 | 375,212 | 530,841 | 484,115 | 321,957 | 164,868 | 63,968 | 18,680 | 4,205 | 571 |

  The count for 5 is derived by subtraction.
- **No S2/S3 record is matched to two S1s.**
- How the labels were produced is Unknown; they were provided by the organisers.

**Missingness and noise** (n4 caches):

| Split | S1 | S2 | S3 |
|---|---|---|---|
| Train | empty address 0%; non-Latin 0%; house number missing 3.5%; legal form present 68% | empty address 3.4%; non-Latin name 9.4%; non-Latin address 9.5%; house number missing 9.3%; legal form 59% | empty address 3.3%; non-Latin name 5.3%; non-Latin address 9.0%; house number missing 9.2%; legal form 58% |
| Test | — | empty address 2.7%; non-Latin name 11.2%; non-Latin address 11.3% | empty address 2.7%; non-Latin name 6.3%; non-Latin address 10.8% |

Duplicate and near-duplicate statistics within S2/S3 were not computed (Not found).

### 6.2 Splits (Verified: `dataset.add_split`, `train.py`)

- **Hash.** h = mix(numeric id, seed 42); fold = (h // 256) mod 5; bucket = h mod 1000. This is **by S1 entity**, never by pair.
- **Roles in M11.**

  | Role | Rule | S1 | Pairs |
  |---|---|---|---|
  | val | fold 0 and bucket < 100 | 44,736 | 2,956,123 (150,040 positive) |
  | es (stage 2) | fold 1 and bucket < 500 | 221,123 | 14,647,891 |
  | train (stage 1) | folds 2–4 and bucket < 300 | 396,588 | 26,311,854 |

- Blocking always searches the **full** train pool, exactly as at test time.

### 6.3 Leakage analysis

| Risk | Finding | Label |
|---|---|---|
| Entity overlap between train and validation | None. Roles are assigned per S1 id. The S2/S3 pool is shared, as it is at test time, but labels are only joined for the S1 in each role | Verified (code) |
| Ground truth used at inference | No. `inference.py` never reads ground truth; only `train.py`, `blocking.py main()`, `evaluate.py` and `decide_eval.py` do | Verified (grep) |
| Candidate-generation leakage | Blocking, name commonness and reverse ranks are label-free statistics of the split's own records. They include the val S1, as test statistics include test S1 | Verified (code) |
| Row-order or ID leakage | The correlation between ID or row position and matches is about 0.0001 (I1). The rank tie bias (B9) depends on row order, but it is label-free and identical in structure on test | Documented |
| Transitive leakage | Stage-2 profile features use p1 of **the same S1's** other candidates. For val and es, p1 is out-of-sample. **In M5–M7**, stage 1 early-stopped on the es fold, so es p1 was mildly in-sample (tree count chosen on es). **Fixed in M11** (`stage1_es_from_train`) | Verified (code) |
| Threshold selection on test | Never. Thresholds are chosen on val fold 0. The leaderboard was used for two uploads and no tuning | Verified |
| Threshold chosen on the same fold it is reported on | About 0.001 optimistic (V6) | Verified |
| Research selection on the val fold | R1–R14 and V3–V7 were run on fold-0 validation pairs. The features adopted in M11 were chosen on the same S1 that report 0.9739, so the reported number carries unquantified selection bias | Inferred risk (untested) |
| Near-duplicates across train and test | Train and test have different ID spaces. Near-duplicate businesses across splits were not checked | Unknown |

---

## 7. Performance history

### 7.1 Full-pipeline validation (44,736 S1; bundles)

| Model | t | Macro F0.5 | Pair P | Pair R | FP | FN | Singleton acc. | Oracle | Stage 1 / 2 trees | Features | Train pairs | Runtime | Peak RAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 dev v1 | 0.75 | 0.9406 | 0.9862 | 0.8798 | Not found | Not found | 0.9432 | 0.9824 | 1491 / — | 49 | 6.65M | 1949 s | 44.09 GB |
| M2 dev v3 | 0.80 | 0.9584 | 0.9916 | 0.9053 | 1,180 | 14,632 | 0.9652 | 0.9824 | 1259 / — | 54 | 6.64M | 873 s | 23.72 GB |
| M5 dev_s2 | 0.75 | 0.9598 | 0.9905 | 0.9139 | 1,360 | 13,310 | 0.9744 | 0.9829 | 1335 / 97 | 54 / 64 | 6.63M | 782 s | 22.5 GB |
| M6 tf30_s2 | 0.775 | 0.9615 | 0.9922 | 0.9145 | 1,107 | 13,213 | 0.9788 | 0.9829 | 2963 / 87 | 54 / 64 | 20.0M | 2195 s | 25.37 GB |
| M7 rev_tf30_s2 | 0.80 | 0.9691 | 0.9938 | 0.9312 | 900 | 10,637 | 0.9828 | 0.9895 | 2778 / 93 | 60 / 70 | 26.3M | 6459 s* | 31.64 GB |
| **M11 final** | **0.80** | **0.9739** | **0.9962** | **0.9401** | **561** | **9,264** | **0.9864** | 0.9895 | 1688 / 169 | 70 / 88 | 26.3M | 2980 s | 44.21 GB |
| M10 dryrun (1% subset; **not comparable**) | 0.65 | 0.9653 | 0.9888 | 0.9349 | 175 | 1,080 | 0.9822 | 0.9892 | 150 / 74 | 70 / 88 | 0.86M | 68 s | 24.21 GB |

\* M7's runtime includes building the reverse cache.

Notes on the table:
- **Pair P/R** are micro precision and recall over pairs.
- **Oracle** is the macro F0.5 if exactly the true pairs among the candidates were predicted, i.e. the blocking ceiling.
- **M11 sweep extras.** The maximum is 0.97472 at t=0.675. With stage 1 alone the score is 0.9723 (t=0.82). Nonsingleton-empty fraction is 0.0087; mean predictions per S1 is 3.26.
- **Not measured anywhere:** AUC, calibration error, B-cubed and micro-F. The official metric is the per-S1 macro F0.5.

### 7.2 Blocking (candidate recall; 44,736 validation S1 against the full pool)

| B1 | B2 | B3 | B4 | B5 | B6 (current) | DMV |
|---|---|---|---|---|---|---|
| 0.9293 / 99.3 | 0.9328 / 52.1 | 0.9490 / 48.3 | 0.9530 / 51.5 | 0.9517 / 50.5 | **0.9706 / 66.1** (p95 107) | 0.9726 / 73.1 |

Each cell is recall / candidates per S1.

**Test:** 72.4 candidates per S1, i.e. 125,457,820 pairs out of a possible 1.73M × 9.97M. That is a reduction ratio of
about 0.999993 (Inferred arithmetic).

### 7.3 Loss decomposition (1 − macro F0.5)

| | Blocking misses | Classifier misses | False merges | Total |
|---|---|---|---|---|
| M5 | 0.0171 | 0.0156 | 0.0076 | 0.0402 |
| M7 | 0.0105 | 0.0153 | 0.0050 | 0.0309 |
| **M11** | **0.0105** | **0.0123** | **0.0033** | **0.0261** |

### 7.4 Leaderboard

| Upload | Model | Val | LB |
|---|---|---|---|
| 1 | M2 | 0.9584 | **0.95099** (rank 603) |
| 2 | M7 | 0.9691 | "same 0.951" (exact value **Not found**) |
| — | M11 | 0.9739 | not submitted |

### 7.5 Which model is which

- **Current baseline** (frozen): M7.
- **Latest verified model:** M11.
- **Best leaderboard:** M2, 0.95099. M7 was about equal, but its exact score is unknown.
- **Best on every validation metric:** M11, except recall at lower thresholds (see its sweep).

### 7.6 Remaining bottlenecks (measured)

1. **Classifier misses (0.0123).**
   - The dominant slices at M7 were empty candidate address (36% of misses) and house-number mismatch (37%). About 53% of misses have p between 0.4 and 0.8 (Documented).
   - M11 reduced house mismatches. A fresh slice breakdown for M11 was **not computed**.
2. **Blocking misses (0.0105).** Unchanged since B6.
   - Of the remaining 4,537 missed pairs: 38% are generic names with an empty address, and about 15% are character typos (B7).
   - Late-file S1s lose recall to the tie bias (B9).
3. **Oracle ceiling 0.9895.** No classifier can exceed it with the current blocking. **A score of 0.999 is not reachable
   with this blocking.**
4. **Validation-to-leaderboard gap of about 0.018.** Unexplained; France is suspected (U1). This is potentially larger than all the validation losses above.

---

## 8. External resources

Full table in ER_RESEARCH_BIBLIOGRAPHY.md. The resources that influenced implemented components are:

| Resource | Influenced | Status |
|---|---|---|
| Sparkly (IDF top-k blocking) | Blocking | adapted concept |
| Papadakis survey, Supervised Meta-blocking | Multi-key channels and rank features | inspired |
| Foursquare Location Matching top solutions (1st place, TheoViel, hobbitlab, Foursquare blog) | Pipeline shape: blocking → GBDT → stage 2 | inspired |
| Splink term frequency | Name-commonness features | inspired |
| Selective EM (Ruan 2025) | Reverse margins (R4) | adapted as features |
| TransClean (2025) | One-hop profile (R9) | adapted as features |
| Waegeman 2014, Instacart F1 optimizer | Expected-F decoding (M3) | rejected |
| Match-Compare-Select, LinkRank | LambdaRank (R7) | rejected |
| Ong 2014 | Missing-data interactions (R2) | rejected |
| GraLMatch | Cross-source corroboration (R13) | rejected |
| EM benchmark re-evaluation | CPU/GBDT design choice | inspired |
| Official challenge repository | Layout and validator | adapted |

---

## 9. Ideas: tested, untested and new

### 9.1 Previously discussed and **tested**

R1–R14, M3, M4, V5, V9 and B6: see §3. Kept: R1, R4, R9, M4, B6. Everything else was rejected or not adopted, with reasons
given in the registry.

### 9.2 Previously discussed and **not tested** (documented in ROADMAP.md or the conversation)

| Idea | Problem it targets | Approach | Motivation | Validation needed | Cost and risk |
|---|---|---|---|---|---|
| B10 hash tie-break | Late-file S1 recall (B9) | `tie_break="hash"` + tie-sharing ranks; rebuild reverse caches; retrain | recall 0.977 vs 0.967 by file position | Full blocking report + retrain; compare with M11 | ~2.5 h rebuild + ~1 h train. Invalidates the current caches (new hash tag). Estimated +0.1–0.2 points (Proposed) |
| P4 threshold calibration from the LB | Test-vs-val shift | Upload the same M11 scores at two thresholds (for example 0.80 and 0.85) | V5 suggests about 0.02 stricter under density | 2 uploads | Uses the daily quota; public leaderboard is a subset |
| P1 France probe | U1 | Upload output_rev with French S1 rows emptied. The score change measures France's contribution | V8, R11 | 1 upload, ~2 min to build | An emptied French row still scores 1 for French singletons; interpret carefully |
| More training data (train_frac 0.5) | Classifier misses | Retrain | M5→M6 +0.0017 at 3× data | Retrain | RAM: 44 GB already at 0.3 |
| Larger stage-2 set | Singletons, borderline pairs | es_frac > 0.5 | V7 +0.0006 per doubling | Retrain | RAM |
| Character 4-gram name key | Typo blocking misses (~15%) | Add a key type | B7 analysis | Blocking report | Hours; est. +0.001 |
| Scale-normalised features for transfer | France transfer | Replace absolute counts with ratios | R12 partial | Transfer proxy | Parked |
| P2 GPU architecture (A100) | Transliteration and name variants; blocking ceiling | multilingual-e5 dense retrieval + FAISS added to the candidates; mDeBERTa/XLM-R cross-encoder on the uncertain band as a stage-2 feature | Ditto, dense-blocking literature | Build and ablate against M11 on fold 0 | Needs A100 access (G7) and approval. Licences are MIT for e5, mDeBERTa and XLM-R (not re-verified). Native script is only 4% of misses (R16), so the gain is uncertain |

### 9.3 New suggestions from this audit (Inferred or Proposed, not previously discussed)

1. **Clean holdout.** Keep an untouched S1 slice (for example fold 0 with bucket ≥ 100) for a final unbiased estimate of M11, because fold-0 bucket < 100 was used for research selection (§6.3).
2. **Slice report for M11.** Rerun the slice breakdown (empty address, house mismatch, native script, singletons) on
   `runs/20260926_123006_final_train/val_pairs.parquet`, to confirm what the new features fixed before planning P2. This is cheap and read-only.
3. **Label-free France diagnostics on M11 test scores** (after P3): the distribution of p, the empty-output rate and the street-conflict rate per country versus val. This checks whether French predictions are under-confident (hurting recall) or over-merging.

---

## 10. Execution guide (summary; the full guide is in ER_REPRODUCIBILITY.md)

Run from `code/business_entity_resolution/src`:

```
pip install -r ../requirements.txt
python train.py --config ../configs/final.json [--model-dir <new dir>]      # 50 min, 44 GB; default overwrites artifacts/models/final
python inference.py --model-dir ../../../artifacts/models/final --out-dir ../../../output_final   # ~2.5 h (Inferred); NOT yet run or approved
```

⚠ Without `--out-dir`, inference overwrites `D:\ml-challenge\output`, which holds the upload-1 files.
Caches in `artifacts/cache/` are required for these timings. A cold rebuild takes several hours.

---

## 11. Status and handover point

**A. Confirmed working.**
- Normalization n4, blocking (forward + reverse), the 70/88-feature two-stage LightGBM, the exclusive + threshold decoder, the output writer and the validator hook.
- Evidence: the V10 trace test (8/8), the V11 dry runs (byte-identical outputs, validator 0 errors), and the M7 test run (validator PASS, `inf_rev.log`).
- M11 training completed (EXIT 0).

**B. Implemented but not fully validated.**
- M11 on the **test split**: never run.
- `tie_break="hash"` (B10).
- Transitive features at test scale (validated only in the V11 dry run on the train split).
- `inference.py --scores/--threshold` re-threshold path (untested end to end).
- SageMaker launcher.
- `decision.expected_f_decode` (rejected, still present).

**C. Experiments completed.**
- B1–B6, M1–M7, M11, R1–R14, V1–V11, S1, S2. Conclusions are in §3.
- The key conclusions:
  - Reverse blocking and competition features matter most.
  - Listwise objectives, hard-negative weighting and missingness models do not help.
  - Density does not explain the leaderboard gap.

**D. Running or interrupted.**
- Nothing is running (the monitor was stopped after M11 finished).
- Interrupted: M9 (rev63_hrt), M11a, I5 (Kaggle packaging).
- Failed to start: M8.

**E. Planned, not started.**
- P3: test inference with M11.
- P4: LB threshold calibration.
- P1: France probe.
- B10: hash tie-break rebuild.
- Final packaging.
- P2: GPU architecture (proposed).

**F. Known bugs and technical debt.**
- The rank tie bias (B9).
- `kt_NS` is a constant feature.
- Exclusivity keeps exact ties.
- Caches are keyed by version, not by content.
- No git.
- README, Documentation_template, EXPERIMENTS.md and ROADMAP.md are stale (U4, U5).
- Validation is optimistic by about 0.001, plus research selection bias.
- The default output directory overwrites upload 1.

**G. Latest verified state.**

| Item | Value |
|---|---|
| Checkpoint | `artifacts/models/final/{model.txt, model2.txt, bundle.json}` (26-09 13:19) |
| Config | `code/business_entity_resolution/configs/final.json` |
| Code | the live `src/` tree as of 26-09 ~12:25. No commit hash exists; the project zip I6 predates the final model |
| Result | Val macro F0.5 0.9739 at t=0.80 (P 0.9962, R 0.9401), oracle 0.9895 |
| Run folder | `artifacts/runs/20260926_123006_final_train` |

**H. Next step planned before this audit** (recoverable from the conversation and ROADMAP.md).
- Run test inference with M11 into `output_final/` (P3). It is **awaiting owner approval**.
- Then upload it. Use the leaderboard feedback (P4, optionally P1) to fix the threshold and the France handling.
- Then package the zip with the final outputs and updated README and Documentation_template.

**I. Missing from the owner.**
- The exact upload-2 score and the current rank.
- Team name and members.
- Approval for P3, P1 and P2.
- A100 access details.

(See ER_KNOWLEDGE_GAPS.md.)

**Prioritised outstanding tasks** (from the documented plans; deadline 27-09 23:59 IST):
1. Get the owner's approval, then run P3, M11 test inference (~2.5 h), into `output_final/` (validator PASS required).
2. Upload M11 at t=0.80. Record the exact score, together with the upload-2 score (G1).
3. Decide between the P4 threshold variant and the P1 France probe with the remaining uploads.
4. Update README.md, Documentation_template.md, EXPERIMENTS.md and ROADMAP.md to M11. Fill in the team information. Build `<team>_submission.zip`.
5. Only if time and approval allow: B10 (tie-break rebuild, ~3.5 h) or P2 (GPU, needs access).

---

## 12. Executive summary

1. **What problem are we solving?**
   - Each clean Source-1 business must be linked to all of its noisy duplicate records in Source 2 and Source 3. The records have name, address and country; the countries are US and India in train, plus France in test only.
   - The score is macro F0.5 per S1, with singletons counted, which rewards precision.
   - Constraints: CPU laptop, no external data, MIT/Apache models ≤ 8B, 5 uploads per day, deadline 27-09 23:59 IST.
2. **What is the exact current architecture?** A4, the M11 model:
   - normalization n4 (anyascii, legal-form classes, address canonicalization, OCR repair);
   - IDF inverted-index blocking with 6 key types, df cap 1000, top 20/10/10 per S1 × source, plus reverse blocking (pool → S1, top 5 added);
   - 70 pair features (rapidfuzz similarities, overlaps, house-change type, forward and reverse ranks, reverse margins) into LightGBM stage 1 (1,688 trees);
   - 10 per-S1 context features and 8 one-hop profile features into LightGBM stage 2 (88 features, 169 trees);
   - exclusive assignment of each record to its best S1, then a global threshold of 0.80.
   - No neural networks and no pretrained weights.
3. **What experiments have we actually performed?** 70 registry entries:
   - 11 blocking or retrieval entries (B0–B10; B7 was analysis only and B10 was implemented but not run);
   - 3 normalization changes (N1–N3);
   - 9 model-training entries (M1, M2, M5–M7, M8 failed to start, M9 was interrupted, M11a aborted, M11) and 2 decision rules (M3, M4); the dry-run model "M10" is recorded as V11;
   - 14 lab research experiments (R1–R14) and 3 desk-evaluated ideas (R15–R17);
   - 11 validation checks (V1–V11); 3 loss decompositions (L1–L3);
   - 3 submission entries (S0–S2: 2 leaderboard uploads plus unsent files);
   - 7 infrastructure entries (I1–I7); 4 proposals or plans (P1–P4).
4. **What are the most important measured results?**
   - Validation macro F0.5 rose from 0.9406 (M1) to **0.9739** (M11: precision 0.9962, recall 0.9401, singleton accuracy 0.9864).
   - Blocking recall rose from 0.929 to 0.9706, with an oracle ceiling of 0.9895.
   - Reverse blocking was the biggest single gain (+0.0076).
   - House-change features, reverse margins and profile features together added about +0.003 in the lab, and the gain held at test-like density.
   - The leaderboard stayed at **0.95099** (M2) and was reported unchanged at about 0.951 for M7. There is a validation-to-leaderboard gap of about 0.018 that test density does not explain (−0.0005).
   - Cross-country transfer drops to 0.847 (US → India), and France shows a 2.5× street-conflict rate, which makes France (15% of test) the prime suspect.
5. **Which research papers and repositories influenced the implementation?**
   - Adapted: Sparkly (blocking), Selective EM (reverse margins), TransClean (profile features).
   - Inspired the pipeline shape: the Foursquare Location Matching top solutions.
   - Inspired features and design: the Papadakis survey, Supervised Meta-blocking, Splink term frequency.
   - Tested and rejected: Waegeman and Instacart (expected-F decoding), LinkRank and Match-Compare-Select (LambdaRank), Ong 2014 (missing data), GraLMatch (cross-source).
   - Adapted for layout and validation: the official challenge repository.
6. **What is currently working, and what is not?**
   - Working: the full pipeline end to end. M7 test outputs passed the validator. M11 is trained and passes the dry run.
   - Not done or not working:
     - M11 has never been run on test.
     - The leaderboard does not reflect validation gains.
     - The hash tie-break is unvalidated.
     - The docs and README are stale.
     - There is no git.
     - SageMaker and Kaggle were never used.
7. **What is the precise next step?**
   - With the owner's approval, run `python inference.py --model-dir ../../../artifacts/models/final --out-dir ../../../output_final` from `src/`. It takes about 2.5 h and must pass the validator.
   - Upload the result and record the exact score.
   - Then use the remaining uploads for threshold or France probes (P4, P1), and package the submission zip.
8. **What information is still missing?**
   - The exact upload-2 leaderboard score and the current rank.
   - The team name and members.
   - The owner's approval for test inference and uploads.
   - France performance (there are no labels).
   - Whether the public leaderboard subset is representative.
   - A100 access details for the proposed GPU path.
   - Raw logs for M3, M4 and V8, and the isolated effects of N1, N2 and B8.
