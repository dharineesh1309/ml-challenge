# ER_KNOWLEDGE_GAPS — missing information, unverified assumptions, open questions

Each item carries a confidence label and the experiment IDs it touches, as used in `ER_EXPERIMENT_REGISTRY.csv`.
Items marked **[OWNER]** can only be answered by the project owner.

## G. Missing information (must be recovered)

| # | Gap | Why it matters | Where it would come from | Label |
|---|---|---|---|---|
| G1 **[OWNER]** | **Exact leaderboard score of upload 2 (S2, M7).** The owner reported only that it showed "the same 0.951". Upload 1 was 0.95099 | Validation improved by +0.0107 (M2 → M7), yet the leaderboard did not move. That is the single most important unexplained observation (see U1) | The leaderboard page or submission history | Unknown |
| G2 **[OWNER]** | Current leaderboard rank after upload 2, and the uploads remaining today (the limit is 5 per day) | Planning P1, P3 and P4 | Leaderboard | Unknown |
| G3 **[OWNER]** | Team name and member list | Required for the zip name `<team>_submission.zip` and for `Documentation_template.md` | Owner | Unknown |
| G4 | Model performance on **France**. France is 15.0% of test S1 (259,452 of 1,732,544) and 0% of train. There are no labels | Probably the largest part of the validation-to-leaderboard gap (U1). It cannot be measured offline | P1 France-empty leaderboard probe, or labelled French pairs (none exist) | Unknown |
| G5 | Whether the public leaderboard subset is representative of the full test set (country mix, size) | Threshold calibration and model choice made from leaderboard feedback | Challenge organisers / problem statement | Unknown |
| G6 | Test-set performance of the **final model M11**. Test inference was never run (P3, not approved) | M11 is only validated offline | Run P3 after owner approval | Unknown |
| G7 **[OWNER]** | A100 server access: `nproc`, `free -g`, `nvidia-smi`, `python3 --version`, connection method, and permission to copy data | Prerequisite for P2 (GPU architecture) | Owner | Unknown |
| G8 | Isolated effects of normalization n2 and n3 (N1, N2) and the k-budget grid table (B8) | They were measured only jointly with other changes, or the table was not saved | Re-run blocking with only that change | Not found |
| G9 | The cause of the first blocking crash (B0, run `20260925_192640`) | Historical only | Not recoverable from artifacts | Unknown |
| G10 | Raw logs for M3 (expected-F decoding) and M4 (exclusivity on M1 scores) | The numbers exist only in EXPERIMENTS.md | Not found on disk. `decide_eval.py` can recompute them from a run folder | Documented but unverified |
| G11 | Raw output of V8 (France street-conflict rate) | The number exists only in EXPERIMENTS.md | The script was run inline and not saved | Documented but unverified |
| G12 | The problem-statement PDF (`D:\6ab10eb3b23ba_student_resource\...problem_statement.pdf`) was read at project start but **not re-read during this audit** | The task requirements in ER_MASTER_HANDOVER.md §2 come from the project notes and the official `student_resource/README.md` / validator | Re-read the PDF | Documented but unverified |

## U. Unresolved contradictions and unexplained observations (not silently resolved)

| # | Observation | Sources | Candidate explanations (none proven) |
|---|---|---|---|
| U1 | Validation → leaderboard: M2 0.9584 → 0.95099 (gap 0.0074); M7 0.9691 → "~0.951" (gap ~0.018). The validation gain did not transfer | experiments.tsv; the owner's reports | (a) **France**: unseen country with a label-free false-merge rate 2.5× that of other countries (V8), and cross-country transfer drops to 0.847 (R11). *Inferred back-of-envelope:* if US and India scored about 0.97 on test, France would need to be about 0.84 to give 0.951 overall, which matches the size of R11's transfer drop. That is consistent, not proven. (b) Test density: ruled out as the main cause, −0.0005 (V1). (c) Threshold or calibration shift (V5 suggests ~0.02 stricter). (d) The public leaderboard subset differs from the full test set (G5). (e) A test-only pipeline defect. V11 ran the dry run on the **train** split; test-only code paths were verified only through the validator PASS and output statistics. (f) The upload-2 score is not really "the same" (G1) |
| U2 | Final model threshold: the bundle chose t=0.80 (0.97390) while the sweep maximum is 0.97472 at t=0.675. A later analysis on a 0.01 grid restricted to 0.50–0.95 reported stage 2 at 0.9737 (t=0.81) | `runs/20260926_123006_final_train/threshold_sweep.tsv`; bundle.json; transcript | Not a bug. `decision.best_threshold` deliberately takes the strictest t within 0.001 of the maximum, and grid spacing changes which t qualifies. Which threshold is better **on test** is Unknown (P4) |
| U3 | M7 metric: 0.9691 in the bundle and experiments.tsv vs 0.9688 in V1/V2 | bundle vs `lab/dmv/baseline_eval.json` | **Threshold rule, not decoder.** With the exclusive decoder, M7 scores 0.9691 at t=0.80 (FP 895) and 0.9688 at t=0.82 (`baseline_eval.json`). The lab sweep used a 0.01 grid, where "strictest t within 0.001 of the max" selects 0.82. The train.py sweep used a 0.025 grid, where it selects 0.80. At the time, train.py swept without the exclusive decoder, a defect since fixed, but at t=0.80 that makes no difference to the M7 score. Both numbers are correct under their own grid |
| U4 | `EXPERIMENTS.md` lists M8 and M9 as "🔄 queued" and has no M10 or M11 rows. `ROADMAP.md` "current best" is still M7 | the files themselves | The documents were last updated about 08:30 on 26-09, before M11. The handover supersedes them. They were **not edited**, because this audit is read-only |
| U5 | `Documentation_template.md` (root) says 0.9691 and 60 features. The README describes dev-era commands and 54 features | the files | Both are stale. M11 has 70 stage-1 and 88 stage-2 features |
| U7 | B1 in EXPERIMENTS.md: "recall 0.929 … 5% of pairs lost". A recall of 0.9293 means 7.1% lost | EXPERIMENTS.md §A; experiments.tsv row 1 | Probably a typo, or the 5% was measured on another basis. The tsv value 0.9293 is Verified |
| U6 | Lab absolute scores (for example the control 0.9625) are below the full-model scores | lab protocol | Expected: half the data and stage 1 only. Only deltas within the lab are comparable. **Never compare lab numbers with bundle numbers** |

## A. Unverified assumptions the pipeline relies on

| # | Assumption | Evidence so far | Label |
|---|---|---|---|
| A1 | Each S2/S3 record belongs to at most one S1 in the test data, as in train (the basis of `decision.exclusive`) | Train ground truth: no shared records (M4) | Verified on train; Inferred for test |
| A2 | The test pool has about 40% unmatched records (the basis of DMV) | Cand/S1 in the DMV universe (73.1) matches test (72.4) | Inferred |
| A3 | France behaves like US/India apart from the address conflict mode | V8 found a difference; nothing else measured | Unknown |
| A4 | The rank tie bias (B9) is harmless to the validation→test comparison because it is the same in both | Measured in both splits | Documented |
| A5 | The hash tie-break (B10) would gain +0.1–0.2 points | ROADMAP estimate only | Proposed |
| A6 | The GPU cross-encoder (P2) would help transliteration and name variants | No experiment. Native script is only 4% of misses (R16) | Proposed |
| A7 | Leakage: none confirmed. Entity-level splits; blocking is label-free; stage-2 inputs are out-of-fold in M11 | See ER_MASTER_HANDOVER.md §6 | Verified by code reading |

## I. Inaccessible or deliberately unused resources

- **Competitor repositories** for this challenge were not opened, on purpose (plagiarism clause).
- The **leaderboard web page** cannot be accessed from here, so every leaderboard number comes from the owner.
- The **AWS accounts** exist, but no SageMaker job ever ran (quota 0).
- The **A100 server** has never been accessed.
- **Kaggle** was never used: the packaging step was interrupted (I5).
- **Earlier conversation context** is available only through the session transcript, which this audit used. Pasted prompts were not saved as files.
- **External URLs were not re-fetched** during this audit (see ER_RESEARCH_BIBLIOGRAPHY.md).

## Q. Questions for the owner

1. What exactly did upload 2 score, and what is the current rank? (G1, G2)
2. May test inference with the final model run (P3, about 2.5 h on the laptop, writing to `output_final/`)?
3. Should one upload go to the France probe (P1) before the final threshold is chosen?
4. What are the team name and members? (G3)
5. Should the GPU architecture (P2) be pursued before the deadline (27-09 23:59 IST), and what are the A100 access details? (G7)
