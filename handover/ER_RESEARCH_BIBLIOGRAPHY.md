# ER_RESEARCH_BIBLIOGRAPHY — external resources

**Source of this list.** Every resource below was cited in the project's own research write-ups during the working
session (strategy pass on 25-09, research pass R1–R14 on 26-09, architecture discussions), which were recovered from the
session transcript. None of them is cited in source-code comments. There are no git remotes.

**Not re-verified.** URLs and bibliographic details were **not re-fetched** during this audit, so titles, authors
and years come from those earlier write-ups. Where a field was not recorded, it says *Not recorded*.

**Status codes**
- **ADAPTED**: a mechanism implemented in our code.
- **INSPIRED**: conceptual inspiration for an experiment.
- **REJECTED**: evaluated and rejected.
- **COLLECTED**: read or collected, but not used.
- **PROPOSED**: named only in a proposal.

**Competitor repositories.** Other teams' repositories for this live challenge appeared in search results, for example
several `github.com/*/Business_Entity_Resolution` repositories. They were **deliberately not opened**, because of the
plagiarism clause and the user's instruction not to inspect competitor challenge repositories. They are not listed as
resources.

## 1. Papers and preprints

| Resource | URL / DOI | Main contribution | Relevance to our problem | What we borrowed / adapted | Status (exp. IDs) |
|---|---|---|---|---|---|
| Paulsen, Govind, Doan — *Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching*, PVLDB 16 (2023) | https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf · repo https://github.com/anhaidgroup/sparkly | Top-k TF/IDF (BM25) blocking over tokens beats complex blockers | Our blocking must search a 10M-record pool per S1 on CPU | IDF-weighted inverted index with top-k per query, df cap on frequent keys (`blocking.build_index`, `query`). The code is our own; nothing was copied from the repo | ADAPTED concept (B1–B6) |
| *Towards Universal Dense Blocking* (arXiv 2404.14831); authors *Not recorded* | https://arxiv.org/pdf/2404.14831 | Dense blocking; notes that Sparkly is weak on transliteration | Motivated the skeleton × address cross key for native-script names | Only the diagnosis; no dense model | INSPIRED (B3); dense retrieval PROPOSED (P2) |
| Papadakis et al. — *Blocking and Filtering Techniques for Entity Resolution: A Survey* (ACM CSUR 2020) | https://arxiv.org/abs/1905.06167 | Taxonomy of blocking, filtering and meta-blocking | Design space for multi-key blocking and budgets | Multi-key (token, pair, cross) blocking with per-channel budgets | INSPIRED (B2–B5, B8) |
| *Generalized Supervised Meta-blocking* (arXiv 2204.08801); authors *Not recorded* | https://arxiv.org/abs/2204.08801 | Learning to prune candidate pairs with block-graph features | Candidate pruning and rank features | Rank / score / key-type features on candidate pairs (`CONTEXT` features); no meta-blocking graph | INSPIRED (M1 features) |
| Li et al. — *Ditto: Deep Entity Matching with Pre-Trained Language Models* (PVLDB 2021) | https://arxiv.org/pdf/2004.00584 · https://github.com/megagonlabs/ditto | Cross-encoder transformer matcher | A reference for a GPU re-ranker | Nothing implemented | COLLECTED; PROPOSED (P2) |
| *Critical re-evaluation of entity-matching benchmarks* (arXiv 2307.01231); authors *Not recorded* | https://arxiv.org/abs/2307.01231 | Shows that many EM benchmarks are easy and that simple baselines are competitive | Justified a CPU-first gradient-boosting approach over deep matchers | The decision to use LightGBM on engineered features | INSPIRED (design choice) |
| *Pre-trained embeddings for entity resolution* (arXiv 2304.12329); authors *Not recorded* | https://arxiv.org/abs/2304.12329 | Evaluates pretrained embeddings for blocking and matching | Option for dense blocking | Nothing implemented | COLLECTED; PROPOSED (P2) |
| Waegeman, Dembczyński, Jachnik, Cheng, Hüllermeier — *On the Bayes-Optimality of F-Measure Maximizers*, JMLR 15 (2014) | https://arxiv.org/abs/1310.4849 | Exact expected-F decoding | Our metric is a per-S1 F0.5 | Per-S1 expected-F0.5 decoder (`decision.expected_f_decode`) | ADAPTED then REJECTED (M3: 0.9389 vs 0.9406) |
| Ong et al. — *Improving record linkage performance in the presence of missing linkage data* (J. Biomed. Inform., 2014) | https://pubmed.ncbi.nlm.nih.gov/24524889/ | Weight redistribution and imputation for missing fields | Empty candidate address is 36% of classifier misses | Missing-address interaction features | INSPIRED then REJECTED (R2, R3) |
| Ruan et al. — *Selective Entity Matching with Contrastive Margin Ranking Loss* (2025, ScienceDirect) | https://www.sciencedirect.com/science/article/pii/S1474034625004318 | Judge a candidate against its competing candidates | Generic empty-address records are claimed by several S1s | Competition implemented as **features**: reverse margins `r_best`, `r_gap_best`, `r_margin12`, `r_ratio`, `r_n` (`blocking.reverse_stats`, `features.REVERSE_MARGINS`). The loss itself was not implemented | INSPIRED → ADAPTED as features (R4 +0.0014, kept in M11) |
| *Match, Compare, or Select? An Investigation of LLMs for Entity Matching* (arXiv 2405.16884); authors *Not recorded* | https://arxiv.org/pdf/2405.16884 | Listwise "select" framing beats pairwise matching | Same as above | Listwise LambdaRank objective | INSPIRED then REJECTED (R7: −0.005) |
| *LinkRank* (arXiv 2607.23610, 2026); authors *Not recorded* | https://arxiv.org/pdf/2607.23610 | LambdaMART for one-to-many set recovery | Our output is a one-to-many set | LambdaRank grouped by S1 | INSPIRED then REJECTED (R7) |
| *TransClean* (arXiv 2506.04006, 2025); authors *Not recorded* | https://arxiv.org/abs/2506.04006 | Transitive consistency between direct and implied pairs | Each S1 has ~3.5 observations, and the confident ones reveal the true house number and name | One-hop S1 profile features (`features.transitive_support`, 8 features, seeds p1 ≥ 0.9) | INSPIRED → ADAPTED (R9 +0.0013, kept in M11); strict variant R14 not adopted |
| *GraLMatch: Matching Groups of Entities with Graphs and Language Models* (EDBT 2025; arXiv 2406.15015); authors *Not recorded* | https://arxiv.org/abs/2406.15015 | Graph cleanup for multi-source company matching | S2↔S3 corroboration | Cross-source corroboration features | INSPIRED then REJECTED (R13: 0) |

## 2. Repositories, write-ups and blogs

| Resource | URL | Main contribution | Relevance | What we borrowed | Status |
|---|---|---|---|---|---|
| Foursquare Location Matching (Kaggle 2022): 1st-place write-up | https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution | Multi-stage candidate generation → GBDT pair model → post-processing for POI matching | Closest public analogue: business name + address matching at scale | Overall pattern: blocking → pair features → LightGBM → a stage-2 model with context of stage-1 scores | INSPIRED (M1, M5 stage 2) |
| TheoViel/kaggle_foursquare (4th place code) | https://github.com/TheoViel/kaggle_foursquare | Candidate generation and GBDT features | Same | Feature ideas (string sims, rank features). No code copied | INSPIRED |
| hobbitlab/Foursquare-Location-Matching | https://github.com/hobbitlab/Foursquare-Location-Matching/blob/main/readme.md | Solution description | Same | Pattern only | INSPIRED |
| Foursquare engineering blog: *Finding the right POI match* | https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/ | Industrial POI matching pipeline | Same | Pattern only | INSPIRED |
| Splink: term-frequency adjustments; comparison and null handling; intro blog | https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html · https://moj-analytical-services.github.io/splink/topic_guides/comparisons/customising_comparisons.html · https://www.robinlinacre.com/introducing_splink/ | Fellegi–Sunter with term-frequency adjustment; null = zero weight | Frequent names are weak evidence | Name-commonness features `cat_df` (kept) and `s1_cat_df` (dropped in V3); IDF weighting in blocking | INSPIRED (M1 features; R5) |
| ing-bank/EntityMatchingModel | https://github.com/ing-bank/EntityMatchingModel | Company-name matching features (string, rank, legal entity, group aggregation) | Company-name matching | Already covered by our ranks, legal-form eq/conflict and stage-2 aggregation | COLLECTED (R15: not run) |
| Instacart Market Basket Analysis: competition summary; Faron's F1 optimizer | https://medium.com/kaggle-blog/instacart-market-basket-analysis-feda2700cded · https://github.com/asagar60/Instacart-Market-Basket-Analysis/blob/main/f1optimization_faron.py | Per-order expected-F1 maximisation | Per-S1 F0.5 decoding | Expected-F decoding idea (with Waegeman) | ADAPTED then REJECTED (M3) |
| AI4Bharat IndicXlit (MIT) | https://github.com/AI4Bharat/IndicXlit | Indic ↔ Latin transliteration models | Native-script S2/S3 names | Nothing. `anyascii` is used instead | COLLECTED; REJECTED as not a failure mode (R16) |
| Official challenge repository (skeleton) | https://github.com/Amazon-ML-Challenge-2026/business-entity-resolution | Expected layout, validator, documentation template | Submission format | Folder layout `code/business_entity_resolution/{src,README.md,requirements.txt}`; official `utils/validate_submission.py` (used unmodified from `student_resource/utils/`) | ADAPTED (layout, validator) |
| AWS SageMaker Python SDK migration notes | https://github.com/aws/sagemaker-python-sdk/blob/master/migration.md | SDK v2/v3 API | SageMaker launcher | API usage in `code/business_entity_resolution/aws/launch.py` | ADAPTED (never ran: I2, I3) |

## 3. Libraries actually used (installed versions verified with `pip show` during this audit)

| Library | Version | Licence (pip metadata) | Used for |
|---|---|---|---|
| polars | 1.42.1 | MIT text, "Copyright (c) 2025 Ritchie Vink" (field shows copyright only) | all dataframes, joins, caching |
| rapidfuzz | 3.14.6 | MIT | `process.cpdist`: ratio, token_set/sort, partial, Jaro-Winkler, Levenshtein |
| lightgbm | 4.7.0 | MIT | stage-1 and stage-2 classifiers |
| anyascii | 0.3.3 | *empty metadata field.* Upstream licence is ISC to my knowledge (Inferred, not verified) | transliteration of non-ASCII names and addresses |
| numpy | 2.3.5 | not re-checked | arrays |
| psutil | 7.2.2 | not re-checked | peak-memory logging |

The challenge's "MIT/Apache, ≤ 8B parameters" rule concerns **models**. No pretrained model is used, and the LightGBM
boosters are trained from scratch.

## 4. Proposed only (P2 GPU architecture, never implemented)

Named in the GPU proposal. URLs were **not recorded and not verified** in the project, so none are given here.

- multilingual-e5: dense multilingual encoder for retrieval.
- FAISS: approximate nearest-neighbour index.
- mDeBERTa-v3 or XLM-R: cross-encoder on the uncertain band (p between about 0.3 and 0.95), output used as a stage-2 feature.

Status: PROPOSED. It needs A100 access details and user approval. See `ER_KNOWLEDGE_GAPS.md` G7.

## 5. User-supplied advice (no URL)

- **Character 3–5-gram TF-IDF retrieval.** Pasted by the user as external advice. Analysed, then parked (B7).
- **Research prompts #1 and #2, the cross-source triangulation prompt and the master research prompt.** These pasted
  documents listed the ideas behind R1–R17, V1–V11 and the triangulation experiment R13. They are only in the
  conversation record and were not saved as files.
