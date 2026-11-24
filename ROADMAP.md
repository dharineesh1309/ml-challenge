# Roadmap — where we are and what comes next

_Last updated: 26 Sep 2026, ~03:15 IST. Deadline: **27 Sep 2026, 23:59 IST**. Uploads: max 5 per day._
Full history of what was tried: [EXPERIMENTS.md](EXPERIMENTS.md).

## Current best

| | |
|---|---|
| Model | `artifacts/models/rev_tf30_s2` (config `configs/rev_tf30_s2.json`) |
| Validation macro F0.5 | **0.9691** (oracle 0.9895) |
| Best leaderboard so far | 0.95099 (sub01, older model) — rank 603; rank 1 ≈ 0.981 |
| Remaining validation loss | classifier misses 0.0153 · blocking misses 0.0105 · false merges 0.0050 |

## STATUS 26-09 08:30 — architecture validated, final training NOT started (awaiting approval)

Final config: `code/business_entity_resolution/configs/final.json` (see EXPERIMENTS.md §V).
Validated lab gain over the baseline stack: ≈ +0.003 macro F0.5 at both normal and test-like density.
Open risk: the val→LB gap (0.969 vs ~0.951) is NOT explained by candidate density (−0.0005); France is the prime suspect.

## Path we are on

```
[done] baseline → error-analysis fixes → stage 2 → 3× data → reverse blocking (0.9691)
[done] research pass (lab harness): KEEP house-change (R1), reverse margins (R4), profile support (R9);
       REJECT missingness interactions/specialist, hard-neg weighting, LambdaRank, cross-source-only,
       feature removal for transfer
   │
   ├─ 🔄 NOW   test inference with rev_tf30_s2  → output_rev/  (ETA ~06:30 IST) → upload 2 (+ stricter-threshold upload 3)
   ├─ ⏭ NEXT  rev63      (ablation, auto-starts)                                  (~50 min)
   ├─ ⏭       rev63_hrt  (rev63 + R1 + R4 + R9 = full validated stack, auto-starts) (~60 min)
   ├─ ⏭       tie-break fix in reverse ranking → rebuild reverse caches (~2.5 h) → retrain best config
   ├─ ⏭       test inference with the best model → upload; threshold from upload-2-vs-3 comparison
   └─ 🏁 27-09 freeze best model → final test run → package zip → methodology doc
```

## Immediate actions

| Who | Action | When |
|---|---|---|
| Me | When `output_rev/` passes the validator → **upload 2** (best model, t=0.80) | ~05:00 IST |
| Me | Same scores at a stricter threshold → **upload 3** (2 min from saved scores) | ~05:05 IST |
| You | Upload 2 and 3; send both scores → sets the final threshold direction | today |
| Me | Train `rev63`, compare with 0.9691 | ~06:00 IST |
| You | Keep the laptop plugged in and awake (battery = 3× slower) | always |

## Decision rules we follow

- A change is **kept only if validation macro F0.5 improves** (noise level ≈ ±0.0007 on 44.7k S1).
- The threshold for test is set from the LB comparison of the best model's scores at two thresholds (upload 2 vs upload 3); move further in whichever direction wins.
- The public LB is a subset of test: use it for direction (threshold, big changes), not for fine-tuning.
- No external data, no competitors' code; models must stay MIT/Apache and ≤ 8B params.

## Candidate next experiments (ranked by expected gain per hour)

| Rank | Experiment | Targets | Expected | Cost |
|---|---|---|---|---|
| 0 | `rev63_hrt`: house change + reverse margins + profile support (validated in lab, R10) | house-mismatch misses, empty-address false merges | ≈ +0.3 pts | 60 min (queued) |
| 0b | Reverse-rank tie-break fix (hash tie-break + tie-sharing ranks) | blocking recall of S1s late in the file (0.967 vs 0.977) | +0.1–0.2 pts | ~2.5 h rebuild + retrain |
| 1 | `rev63`: S1-side name commonness + channel agreement | empty-address classifier misses | lab: inconclusive (+0.0003) | 50 min (queued) |
| 2 | Threshold calibration from LB (upload 2 vs 3, same best-model scores) | test/val shift (−0.007) | up to +0.5 pts on LB | 2 min per variant |
| 3 | Larger stage-2 training set (bigger early-stop fold) | singletons, borderline pairs (53% of misses have p 0.4–0.8) | +0.05–0.15 | 1 h |
| 4 | More training data (`train_frac` 0.5) | classifier misses | +0.05–0.1 | 1.5 h, RAM-limited |
| 5 | Character 4-gram name key in the IDF index | typo-only blocking misses (~15% of 4,537) | ≈ +0.1 | 2–3 h ⏸ |
| 6 | GPU transformer re-ranker on uncertain pairs (A100 server, if available) | transliteration / name variants | uncertain | 3–4 h + server |

## Final-day checklist (27 Sep)

- [ ] Pick final model by validation F0.5 (+ LB threshold calibration)
- [ ] Final test run → `output/matching_results.tsv` + `output/candidate_pairs.tsv`, validator PASS with `--check-ids`
- [ ] Upload final `matching_results.tsv`
- [ ] Build `<team_name>_submission.zip`: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md`
- [ ] Fill team name / members / final LB score in `Documentation_template.md`
- [ ] Update README final-config commands (`configs/rev_tf30_s2.json` or successor)
