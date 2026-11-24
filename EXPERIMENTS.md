# Experiment Log — Business Entity Resolution

Every experiment we ran, what changed, what we measured, and whether we kept it.
All validation numbers are the **official macro F0.5** on the same 44,736 held-out S1 entities
(fold 0, full pipeline: blocking → features → model → decision), unless stated otherwise.
Machine-written raw rows live in `artifacts/experiments.tsv`; per-run details in `artifacts/runs/<timestamp>_<name>_*`.

**Legend:** ✅ kept · ❌ rejected · ⏸ parked · 🔄 running

---

## A. Blocking experiments (candidate generation)

Metric: blocking recall = true pairs in the candidate set / all true pairs of the query S1s (44,736 validation S1, full 10.3M pool).

| # | Date (IST) | Change | Recall | Cand/S1 | Runtime | Decision / why |
|---|---|---|---|---|---|---|
| B1 | 25-09 19:32 | 4 key types (name token, address word, house×word, concat name), single top-50 per source, cap 5000 | 0.929 | 99 | 203 s | Baseline. 5% of pairs lost; recall@1 only 0.43 |
| B2 | 25-09 19:46 | Name/address/combined **channels**, + name-pair key, + skeleton key, generic-word stripping (norm n2) | 0.933 | 52 | 568 s | ✅ same recall at **half the candidates**. Skeleton key alone: 0 unique recoveries → dropped later |
| B3 | 25-09 19:56 | + **name-skeleton × address-word cross key** (`mdrn\|jaipur`), − skeleton unigram | 0.949 | 48 | 514 s | ✅ native-script recall 0.71 → 0.85 |
| B4 | 25-09 21:07 | norm n3 (ordinals, state codes) + df cap 2000 | 0.953 | 52 | 174 s | ✅ recall up, faster |
| B5 | 25-09 21:09 | df cap **1000** | 0.952 | 51 | 129 s | ✅ chosen: −0.001 recall for ~3.5× faster query (test inference 855 → ~190 s / 100k S1) |
| B6 | 26-09 00:42 | **Reverse blocking**: every pool record queries an S1 index; add pairs where S1 is in its top-5; reverse ranks as features | **0.971** | 66 | +65 min/split (cached) | ✅ recovered 37% of misses. Probe: 97.9% of true pairs have the S1 as the record's reverse top-1 |
| B7 | 26-09 03:00 | Character 3–5-gram TF-IDF name/address retrieval (external suggestion) | — | — | — | ⏸ parked: only ~15% of the remaining 4,537 misses are char-level typos → est. +0.001 F0.5 for 10+ GB / hours of compute. 38% of misses are empty-address generic names (ambiguous, not a retrieval gap) |

Budget grid (B3–B5) chose `k_name=20, k_addr=10, k_comb=10` per source (Pareto frontier of recall vs candidates).

---

## B. Model / feature / decision experiments

| # | Date (IST) | Model | Change vs previous | Train pairs | Thr | **Val macro F0.5** | Pair P / R | Singleton acc | Oracle | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 25-09 20:29 | `dev` (v1) | Baseline LightGBM (lr 0.05, 127 leaves), 49 features, norm n2, cap 5000 | 6.6 M | 0.75 | 0.9406 | 0.986 / 0.880 | 0.943 | 0.9824 | Baseline |
| M2 | 25-09 21:24 | `dev` (v3) | Error-analysis fixes: **legal-form class + conflict features**, house-number prefix + edit distance, ordinal words, state codes; lr 0.1 / 63 leaves; cap 1000 | 6.6 M | 0.80 | **0.9584** | 0.992 / 0.905 | 0.965 | 0.9824 | ✅ +1.78 pts. **LB 0.95099** (sub01) |
| M3 | 25-09 22:59 | — | Decision: per-S1 expected-F0.5 decoding (on M1 scores) | — | — | 0.9389 | 0.981 / 0.887 | 0.790 | — | ❌ worse than global threshold (0.9406); hurts singletons |
| M4 | 25-09 22:59 | — | Decision: exclusivity (record → its best S1) (on M1 scores) | — | 0.75 | 0.9406 | — | — | — | ✅ kept (neutral on val subset, principled: GT has 0 shared records; matters more on full test) |
| M5 | 25-09 23:07 | `dev_s2` | norm n4 (OCR digit→letter, `lnc`, `& Co` class) + **stage-2 model** (per-S1 context of stage-1 score) | 6.6 M | 0.75 | 0.9598 | 0.991 / 0.914 | 0.974 | 0.9829 | ✅ +0.14 (n4 alone 0.9591, stage 2 +0.07; singleton 0.966→0.974). sub02/sub03 |
| M6 | 26-09 00:41 | `tf30_s2` | 3× training data (`train_frac` 0.1 → 0.3) | 20.0 M | 0.775 | 0.9615 | 0.992 / 0.915 | 0.979 | 0.9829 | ✅ +0.17 |
| M7 | 26-09 02:29 | `rev_tf30_s2` | + reverse blocking candidates & reverse-rank features (60 features) | 26.3 M | 0.80 | **0.9691** | 0.994 / 0.931 | 0.983 | 0.9895 | ✅ **+0.76** — best so far. `rrank_c` is the #1 feature (~8× the next) |
| M8 | queued | `rev63` | + S1-side name commonness (`a/b_s1_cat_df`) + retrieval-agreement count (`n_channels`) → 63 features | 26.3 M | tbd | 🔄 | | | | Targets empty-address misses (36% of classifier misses) |
| M9 | queued | `rev63_hrt` | rev63 + house change type (R1) + reverse margins (R4) + transitive/profile support in stage 2 (R9) | 26.3 M | tbd | 🔄 | | | | Harness predicts ≈ +0.3 pts over the stage-2 control |

### Loss decomposition (1 − val macro F0.5)

| Model | Blocking misses | Classifier misses | False merges | Total |
|---|---|---|---|---|
| M5 `dev_s2` | 0.0171 | 0.0156 | 0.0076 | 0.0402 |
| M7 `rev_tf30_s2` | 0.0105 | **0.0153** | 0.0050 | 0.0309 |

→ After M7 the **classifier** (6,100 missed true pairs, 36% with empty candidate address) is the largest loss.

---

## R. Research pass (isolated lab harness, `lab/`)

Protocol: the saved validation pairs of `rev_tf30_s2` (2.96M pairs, 44,736 S1), split by S1 into two halves;
train on one, score the other, swap; official macro F0.5 with the pipeline's decision (exclusive + swept
threshold). Absolute scores are below the full model (half the data, one stage); **deltas** are what count.
Noise floor from control seeds: ±0.0004 (stage-1 control), ±0.0002 (stage-2 control). Raw rows: `lab/results.jsonl`.

| # | Idea (source → mechanism) | Our failure slice | Control | Result (2 seeds) | Δ | Decision |
|---|---|---|---|---|---|---|
| R1 | House-number change type (numeric diff, digits changed, position, transposition) | house mismatch = 37% of misses; noise = random digit, look-alikes = nearby numbers | 0.9625 | 0.9649 / 0.9646 | **+0.0023** | ✅ KEEP (opt-in `HOUSE_CHANGE`) |
| R2 | Ong et al. 2014 weight redistribution → explicit missing-address interactions | empty address = 36% of misses | 0.9625 | 0.9623 | 0 | ❌ trees already condition on missingness |
| R3 | Empty-address specialist model | same | 0.9625 | 0.9625 | 0 | ❌ trades FN for FP; less data |
| R4 | Selective/listwise EM (Ruan 2025) → reverse-competition margins (this S1 vs the record's best/2nd S1) | ambiguous empty-address names | 0.9625 | 0.9640 / 0.9638 | **+0.0014** | ✅ KEEP (opt-in `REVERSE_MARGINS`); empty-addr FP 170→70-85 |
| R5 | S1-side name commonness + channel agreement (= rev63 additions) | empty address | 0.9625 | 0.9628 | +0.0003 | ⚪ INCONCLUSIVE (overlaps R4); full-scale answer from rev63 |
| R6 | Hard-negative up-weighting ×3 (p1 0.3–0.95) | p 0.4–0.8 band | 0.9625 | 0.9624 | 0 | ❌ only moves the threshold; negatives are already blocking-hard |
| R7 | LambdaRank grouped by S1 (listwise objective) | candidate competition | 0.9625 | 0.9575 | **−0.005** | ❌ loses cross-S1 calibration (singletons 0.96→0.94); competition works as *features*, not objective |
| R8 | R1 + R4 combined | both | 0.9625 | 0.9664 / 0.9664 | **+0.0039** | ✅ additive → queued full run `rev63_hr` |
| R9 | TransClean 2025 transitive consistency / one-hop S1 profile: candidate vs the S1's other confident matches (p1≥0.9) | house mismatch (the S1's other observations reveal the true number) | 0.9688 (stage-2 ctl) | 0.9703 / 0.9700 | **+0.0013** | ✅ KEEP pending R10; house-diff misses −24% |
| R10 | stage2 + R1 + R4 (+ R9) | additivity check | 0.9688 (stage-2 ctl) | 0.9710 / 0.9709 → **+R9: 0.9718 / 0.9716** | **+0.0029** total | ✅ all three additive → full run `rev63_hrt` |
| R11 | Cross-country transfer diagnostic (France proxy) | unseen country | — | US→US 0.970, **US→India 0.847**, India→India 0.953, India→US 0.947 | | ⚠ large, asymmetric transfer drop → R12 |
| R12 | Drop feature groups to improve transfer (IDF scores / counts / abs overlaps / script / all scale) | France robustness | 0.847 US→India | best: −counts 0.864, −all_scale 0.863; but in-country −0.003…−0.005 and India→US −0.006 | | ❌ removal closes ≤1.6 of 10.6 transfer points at a certain in-country cost; ⏸ scale-normalised variants. Best hedge = train on both countries, no country feature (as now) |
| R13 | GraLMatch / TransClean → **cross-source corroboration only** (S2↔S3 among plausibly-same-S1 candidates: top-3 in source & p1≥0.5; support / contradiction / opposite-source score) | empty address, false merges | 0.9688 (stage-2 ctl) | 0.9692 / 0.9686; on full stack 0.9719 / 0.9716 (vs 0.9718 / 0.9716) | 0 | ❌ REJECT. Flip analysis: 678 fixed vs 535 broken with the *same* share of cross-source support (48% vs 47%) and contradiction (2.2% vs 2.4%) → model variance, no new information; gated opposite-source records are already well-scored, and R9's profile + stage-2 context carry what they know |
| R14 | One-hop profile with **strict** seeds (p1≥0.95, 3 features), separate from R13 | house mismatch | 0.9688 | 0.9696 / 0.9698 | +0.0010 | ⚪ same mechanism as R9 (p1≥0.9, 8 features, +0.0013) → keep R9's version; the signal is the S1's confident profile, not cross-source similarity per se |
| — | ING EMM (rank, legal-entity, group features) | — | — | not run | | already captured (ranks, legal-form eq/conflict, stage-2 aggregation) |
| — | Multilingual / transliteration (#8) | native script = 4% of misses, recall above average | — | not run | | ❌ not a current failure mode |
| — | Source-aware (#4) | S2/S3 recall 0.961 / 0.958 | — | not run | | ⏸ no gap to fix; `src` already a feature |
| — | Char n-gram / dense blocking | 15% of remaining blocking misses are char-typo | — | not run | | ⏸ est. +0.001 for hours of compute |

**Bug found during research:** reverse-rank ties were broken by S1 row order, so S1s early in the file win
ties: reverse adds per S1 fall 97 → 20 (test) and 89 → 24 (train) across the file; validation blocking recall
0.977 (first 20% of S1) vs 0.967 (last 20%). Consistent between val and test (not a leak), but costs recall.
Fix = hash-based tie-break + tie-sharing ranks; needs reverse-cache rebuild (~2.5 h). Scheduled after `rev63_hr`.

---

## V. Architecture validation phase (26-09, no full training)

Baseline frozen in `artifacts/baseline_best/` (models, config, code snapshot, md5s, `BASELINE.json`).
Density-matched validation (DMV): hide 25% of NON-validation S1 (20% of all) so their matches become orphaned
look-alikes; validation S1 + ground truth untouched. Pool unmatched share 26% → 40.8% (test ≈ 40%);
candidates/S1 66.1 → 73.1 (actual test 72.4). Pairs rebuilt with the current code (`lab/dmv_build.py`).

| # | Check / experiment | Result | Decision |
|---|---|---|---|
| V1 | Baseline at normal vs test-like density (frozen model, exclusive decoder) | 0.9688 → **0.9683** (−0.0005); FP 829 → 990; best t 0.82 both | density explains ≈0.0005 of the ~0.018 val→LB gap → **not the main cause** |
| V2 | Stage 1 vs Stage 2 of the baseline (exclusive decoder) | 0.9687 vs 0.9688 | stage 2 adds ~0 at full scale (only singletons 0.977→0.983) → give stage 2 more OOF data (V7) |
| V3 | Stack ablations at test-like density (2 seeds; control 0.9685) | +rev63 0.9686, +house 0.9699/0.9698, +revm 0.9695/0.9688, +trans 0.9698/0.9696, **full 0.9713/0.9714**, full−rev63 0.9714/0.9717 | gains hold at test density (+0.0029); **drop rev63 features** (no value) |
| V4 | Same at normal density (control 0.9686/0.9687) | full 0.9717/0.9721, full−rev63 0.9720/0.9719 | consistent (+0.0030) |
| V5 | Density-matched TRAINING (train normal → eval DMV vs train DMV → eval DMV) | 0.9711/0.9711 vs 0.9713/0.9714; best t 0.85-0.86 vs 0.83 | not needed; at most threshold ≈ +0.02 stricter |
| V6 | Threshold selection bias (pick t on one half, apply to other) | cost 0.0003–0.0018 (≈0.001); curve flat 0.6–0.85 | reported val ≈0.001 optimistic; low threshold sensitivity |
| V7 | Stage-2 learning curve (25/50/100% of fitting S1) | 0.9701 / 0.9707 / 0.9713 (+0.0006 per doubling) | use `es_frac=0.5` (≈5× stage-2 data, OOF-clean) |
| V8 | France street-conflict (label-free): accepted pairs where each address has a rare word the other lacks | FR 3.5% vs US 1.5%, IN 1.3% | France-specific false-merge mode identified |
| V9 | IDF-weighted address + conflict features (country-agnostic fix for V8) | in-domain neutral (±0.0001); transfer proxy −0.0009/−0.0004 | ❌ not integrated (benefit unverifiable without French labels) |
| V10 | Controlled trace universe (13 true pairs, 8 case types) through real pipeline + baseline models | recall 13/13, schema OK, determinism OK, exclusivity OK, validator PASS; found 0/0 NaN in `score_*_rel` → fixed (null) | ✅ |
| V11 | Dry run of real `train.py` + `inference.py --max-s1 5000` (twice) | train OK (68 s); outputs byte-identical; validator rules 0 errors; matches ⊆ candidates; ≤1 S1 per record | ✅ |

Code fixes in this phase: train-time threshold sweep now uses the inference decoder (exclusive); opt-in
`stage1_es_from_train`, `es_frac`, `drop_features`; `--max-s1` dry-run mode; null instead of 0/0 in relative scores.

---

## C. Leaderboard submissions

| Sub | Date | File | Model | Thr | Val F0.5 | **LB score** | Rank | Notes |
|---|---|---|---|---|---|---|---|---|
| 01 | 25-09 22:55 | `submissions/sub01_v3_t080` | M2 `dev` | 0.80 | 0.9584 | **0.95099** | 603 | LB ~0.007 below val: test pool has ~1.6× more unmatched look-alikes per S1 |
| 02 | 26-09 ~05:00 | `output_rev/` | M7 `rev_tf30_s2` | 0.80 | 0.9691 | _fill in_ | | best model; test inference running |
| 03 | 26-09 | `output_rev_t0xx/` (from saved scores) | M7 `rev_tf30_s2` | stricter (≈0.88) | — | _fill in_ | | same scores as 02, stricter threshold → threshold direction on test |
| — | not uploaded | `submissions/sub02_v4s2_t075`, `sub03_v4s2_t088` (`output_s2*/`) | M5 `dev_s2` | 0.75 / 0.88 | 0.9598 | — | | superseded by M7 before upload; kept on disk |

---

## D. Infrastructure / other trials

| # | What | Result | Decision |
|---|---|---|---|
| I1 | Leak check: row order / ID numbers vs matches | corr ≈ 0.0001, entity rows spread over the whole file | No leak; gap to top teams is modelling |
| I2 | SageMaker, account `…807434` (us-east-1) | Training quota 0 on all 14 instance types tried; $0 spent | ❌ stopped |
| I3 | SageMaker, profile `ml-challenge-AD` (eu-north-1, free tier) | Quota 0 on all 10 types; free tier only allows 4-core/16 GB instances (pipeline needs ~45 GB) | ❌ stopped; launcher kept in `code/.../aws/` |
| I4 | Laptop on battery | Chunks 3× slower (182 s → 550 s) | Keep plugged in |
