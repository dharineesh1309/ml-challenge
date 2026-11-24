# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We link each Source-1 business to its Source-2/3 records with a **blocking + gradient-boosted matcher** pipeline that runs on CPU, uses only the provided data and no pretrained or external models. Candidates come from a search-engine-style **IDF-weighted inverted index over six typed keys, queried in both directions** (S1 → pool and pool → S1), which lifts blocking recall to 0.971 at ~66 candidates per S1. A two-stage LightGBM scores 60 similarity/context features per pair; decisions use an exclusive best-S1 assignment and a threshold tuned directly on the official macro F0.5. Validation macro F0.5: **0.9691** (held-out S1 entities, full pipeline).

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the training data (sampled GT pairs and full-file statistics):

- **Country is a safe partition:** 100% of true pairs share `country`. We treat it as an open set of string codes (France appears only in test).
- **Each S2/S3 record belongs to at most one S1** (0 of 7.64M matched IDs repeat) → justifies an exclusive assignment.
- **Names are heavily reused:** an exact normalized-name key matched ~43 non-matching pool records per S1 (5,683 true vs 143,902 false hits). Name agreement alone is weak evidence; **address agreement is the main precision signal**.
- **Addresses:** 95% of true pairs share ≥2 address tokens, 82% share a house number; postal codes are essentially absent (<0.1%) → no postal-code blocking. S1 uses `NC`-style state codes, S3 full names (`North Carolina`), S2 upper-case with abbreviations (`CT`, `AVE`, even `SAINT` for St).
- **Noise patterns:** ~7% of candidate names / ~9% of addresses in native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam); OCR-style swaps (`F0x`, `lnc`, `5ervices`); duplicated words; junk prefixes (`--`, `<<`, `[ ]`); names replaced by domains/hashtags/trade names; ~4% empty candidate addresses; house numbers truncated (`2108`→`210`, `3710`→`710`); ordinal words (`Twelfth` vs `12th`).
- **Hard negatives:** same name + same address with a *different legal form* (`Swastik It Pvt Ltd` vs `Swastik It LLP`) is frequently a non-match.
- **Singletons:** 5.6% of training S1 entities have no match.
- **Test differs:** ~5.76 S2/S3 records per S1 vs 4.68 in train (≈1.6× more unmatched look-alikes), which makes thresholds tuned on train slightly permissive on test.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + constrained decision (hybrid rules/ML).  
**Core Innovation:** (1) bidirectional IDF-weighted blocking with a *name-skeleton × address-word* cross key and separate name/address ranking channels; (2) reverse-rank features ("is this S1 the pool record's best S1?"); (3) name-commonness (term-frequency) and legal-form-conflict features targeting the dataset's hard negatives; (4) every decision tuned on the official per-entity macro F0.5, measured on the full pipeline.

```
raw TSV -> normalization (multi-representation) -> bidirectional blocking -> 60 pair features
        -> LightGBM stage 1 -> per-S1 context -> LightGBM stage 2 -> exclusive best-S1 + threshold
        -> matching_results.tsv (+ candidate_pairs.tsv = exactly the pairs scored)
```

Deterministic components: normalization, blocking, features, exclusivity. Learned: the two LightGBM models. Tuned on validation: the threshold.

**Normalization** keeps raw fields and derives several representations per record: ASCII-folded / transliterated text (anyascii; Indic scripts → Latin), cleaned name tokens (junk and duplicate removal, domain → body, OCR digit→letter repair inside mixed tokens), a **legal-form class** (pvt_ltd, llp, llc, inc, & co, sarl, …) removed from the name but kept as a feature, a **name signature** without generic words that sources add/drop (measured on 964k true pairs: `center`, `services`, `group`, `shri`, …), a **consonant skeleton** (`Global`/`globl` → `glbl`), canonical address tokens (street-type and direction abbreviations incl. French `r`/`av`/`bd`, US/Indian state names → codes, ordinal words → numbers), all address numbers and the house number.

**Validation design:** split by S1 entity (stable hash of the ID): training folds, a separate early-stopping fold, and a validation fold used only for thresholds and reporting. Blocking always searches the full S2/S3 pool, as at test time. We implemented the official metric exactly (reproduces the statement's 0.714 example and singleton rules) and evaluate the whole pipeline, including an "oracle" (perfect classifier on our candidates) to separate blocking loss from classification loss.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (all scoped to the record's country; each key's weight is its IDF over the pool):

  | key | example | recovers |
  |---|---|---|
  | name signature token | `colombier` | reordered names, typos elsewhere |
  | name token pair | `life\|baptist` | names made of common words |
  | concatenated name | `maurewilliamscolombier` | domain / hashtag names |
  | address word | `wynne` | reordered / abbreviated addresses |
  | house number × address word | `3908\|wynne` | most selective address evidence |
  | name skeleton × address word | `mdrn\|jaipur` | common name + common city; native-script names |

  Keys held by more than 1,000 pool records are dropped (they carry almost no information and dominated join cost; recall cost −0.001, 3.5× faster).
- **Ranking:** shared keys add IDF to a name score and/or an address score. Per S1 and per source, we keep the union of the top-20 by name, top-10 by address and top-10 by combined score, so empty-address and native-script matches are not crowded out by address-rich look-alikes.
- **Reverse pass:** every pool record also queries an index over S1 keys; pairs where the S1 is among the record's top-5 S1s are added, and reverse ranks (top-10) become features. This recovered 37% of forward-blocking misses.
- **Candidate pairs generated:** ~66 per S1 (validation: 2.96M pairs for 44,736 S1); test: see `output/candidate_pairs.tsv`. Reduction ratio > 0.999.
- **How we ensured true matches were not lost:** a dedicated blocking report on held-out S1s (recall overall and by country / source / script / empty address, recall-vs-budget Pareto grid, per-key-type unique recoveries, and a dump of every miss with its reason: no shared key / key above frequency cap / below top-k). Each blocking change was accepted only if it raised measured recall per candidate. Recall progression: 0.929 → 0.933 (channels) → 0.950 (cross key) → 0.953 (normalization fixes) → **0.971** (reverse pass).

---

## 4. Matching Model

**Features used (60 per pair; no country feature, so unseen countries are handled identically):**
- **Name features:** Levenshtein ratio, token-set / token-sort / partial ratios, Jaro-Winkler on the signature, skeleton token-set ratio, signature token overlap and Jaccard, exact / contained concatenated name, token counts.
- **Address features:** Levenshtein, token-set and partial ratios on canonical addresses, shared-word count and Jaccard, number-set overlap, house number equal / suffix-contained (`710`⊂`3710`) / prefix (`210` vs `2108`) / edit distance.
- **Legal form:** equal, conflicting, missing on one side.
- **Name commonness (term-frequency):** number of pool records in the country with the identical name signature, for both sides.
- **Data quality / script:** empty address flags, non-Latin script flags.
- **Blocking context:** name / address / combined IDF scores, ranks, score relative to the S1's best candidate, number of candidates, which key types matched, reverse ranks and reverse score, "added by reverse pass" flag.
- **Stage-2 per-S1 context** (from stage-1 probabilities): rank within the S1 and within the S1×source, S1 max / second-best / sum, gap to the best, count above 0.5.

**Model type:** LightGBM (MIT licence, CPU). Stage 1: binary objective, learning rate 0.1, 63 leaves, early stopping (≈2,800 trees), trained on 26.3M pairs from 397k training S1 (5.1% positive). Negatives are the blocking output itself, i.e. realistic hard negatives, never random pairs. Stage 2: same model family on stage-1 features + per-S1 context, trained on the early-stopping fold where stage-1 scores are out-of-sample (no leakage).

**Threshold selection method:** sweep 0.05–0.95 on the validation fold, maximizing the official macro F0.5 over all validation S1 (singletons included); near-ties resolved toward the stricter threshold. Before thresholding, each S2/S3 record is kept only for the S1 with its highest probability (exclusive assignment, justified by the training GT). Evaluated and rejected: per-S1 expected-F0.5 decoding (0.9389 vs 0.9406 for the global threshold; it hurt singletons).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation (44,736 held-out S1, full pipeline): 0.9691** at threshold 0.80 — pair precision 0.9938, pair recall 0.9312, singleton accuracy 0.9828, non-singletons left empty 0.97%; oracle within candidates 0.9895.
- **Most important features (LightGBM gain):** the reverse combined rank (`rrank_c`, ~8× the next feature), best reverse rank, house-number edit distance, reverse address rank, number-set Jaccard, name ratio, address token-set ratio. Stage 2 relies on the stage-1 score plus the reverse rank and the gap to the S1's best candidate.

  | step | val macro F0.5 |
  |---|---|
  | baseline (4 key types, 49 features) | 0.9406 |
  | + legal-form / house-prefix / ordinal / state fixes | 0.9584 |
  | + OCR fixes, `& Co` class, stage-2 model | 0.9598 |
  | + 3× training data | 0.9615 |
  | + reverse blocking & reverse-rank features | **0.9691** |

  Leaderboard: first submission (0.9584 model) scored 0.95099 — the test pool contains more unmatched look-alikes, so later submissions calibrate the threshold on the leaderboard. [final leaderboard score: fill in]

- **Loss decomposition (earlier model, 0.9598):** blocking misses 0.017, classifier misses 0.016, false merges 0.008; singletons 0.0014. This analysis motivated the reverse pass.
- **Common false positives (wrong merges):** records that are near-exact copies of the S1 (often the *cleanest* variant) but are linked to no S1 in the ground truth, frequently with an added or changed legal form (`Grand Classic Transition` vs `… LLC`, `Innovative Sangh` vs `… Private Limited`, `Misha … Inc` vs `… Group`).
- **Common false negatives (missed matches):** candidates with an empty address and a generic name, names replaced by unrelated trade names where only the address links them, Wisconsin "Town of X" addresses where the other source uses the mailing city, and legal-suffix swaps on true pairs with no address.

---

## 6. Conclusion

A carefully measured blocking stage mattered more than model complexity: most of the gap to a perfect score was true pairs that never reached the classifier, and a reverse (pool → S1) pass recovered a large share of them. Features aimed at this dataset's specific hard negatives (legal-form conflicts, name commonness, truncated house numbers) and decisions tuned on the exact per-entity metric gave the remaining gains, all on CPU with a small, fully reproducible LightGBM model.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — see its `README.md` for exact commands.

| file | role |
|---|---|
| `src/config.py` | dataclass config + JSON overlay; CLI / SageMaker env path resolution |
| `src/data_io.py` | TSV readers (UTF-8, quoting disabled), parquet cache |
| `src/normalize.py` | normalization and representations |
| `src/dataset.py` | stable split by S1 entity |
| `src/blocking.py` | typed keys, IDF index, 3-channel top-k, reverse pass, recall report |
| `src/features.py` | pair and per-S1 context features |
| `src/train.py` | **entry point:** blocking → features → LightGBM (2 stages) → threshold → model bundle |
| `src/inference.py` | **entry point:** bundle → blocking → features → scores → decision → both TSVs → official validator |
| `src/decision.py`, `src/metrics.py`, `src/submission.py` | decision rules, official metric, TSV writers |
| `src/evaluate.py`, `src/decide_eval.py` | error analysis tools |
| `configs/*.json` | experiment configurations (final: `rev_tf30_s2.json`) |
| `aws/` | optional SageMaker launcher running the same entry points |

Reproduce: `python src/train.py --config configs/rev_tf30_s2.json` then `python src/inference.py --model-dir <bundle>`; outputs `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs `utils/validate_submission.py`. All randomness is seeded; every run logs its config and metrics to `artifacts/runs/` and `artifacts/experiments.tsv`.

### B. Additional Results

- Blocking recall by slice (forward pass only, frequency cap 1,000): US 0.971, India 0.922, empty-address candidates 0.69; native-script candidates 0.85 (measured with the cross key at cap 5,000).
- Runtime (16-core laptop, 64 GB): normalization + index ≈ 10 min per split (cached), reverse pass ≈ 65 min per split (cached), training ≈ 1.5 h at the final data size, test inference ≈ 2 h.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
