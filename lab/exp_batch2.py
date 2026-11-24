"""Batch 2 of isolated ablations (each = control + ONE idea), same harness/protocol as exp_house.

E-MISS-X   missing-address interactions (Ong et al. weight redistribution, as explicit features)
E-MISS-SP  empty-address specialist model (its OOF replaces the control's on empty-address rows)
E-REVM     reverse relative evidence: this S1's reverse score vs the record's best / 2nd-best S1
E-S1CAT    rev63 additions (S1-side name commonness + retrieval-channel agreement), for attribution
E-HARDNEG  up-weight hard negatives (label 0 with stage-1 p in 0.3-0.95) x3
E-RANK     LambdaRank objective grouped by S1 (same features)
"""
import sys, time
sys.path.insert(0, "D:/ml-challenge/lab")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import polars as pl
from harness import BASE_FEATURES, BUNDLE, CACHE, PARAMS, evaluate, fit_predict, load, log
import glob

t0 = time.time()
v, ids, truth, _ = load()
only = set(sys.argv[1:])            # optionally run a subset: python exp_batch2.py E-REVM E-RANK
run = lambda k: not only or k in only

# ---- E-MISS-X: explicit interactions with candidate-address missingness
m = pl.col("b_addr_empty").cast(pl.Float32)
v = v.with_columns(mx_name_tset=m * pl.col("name_tset"), mx_sig_jw=m * pl.col("sig_jw"),
                   mx_cat_eq=m * pl.col("cat_eq").cast(pl.Float32), mx_rrank=m * pl.col("rrank_best"),
                   mx_catdf=m * pl.col("b_cat_df"), mx_score_n_rel=m * pl.col("score_n_rel"))
MX = ["mx_name_tset", "mx_sig_jw", "mx_cat_eq", "mx_rrank", "mx_catdf", "mx_score_n_rel"]

# ---- E-S1CAT: rev63 features computed from caches
s1n = pl.scan_parquet(CACHE + "norm_n4_train_s1.parquet").select("entity_id", "country", "name_cat").collect()
s1cnt = s1n.group_by("country", "name_cat").agg(s1_cat_df=pl.len())
a = s1n.filter(pl.col("entity_id").is_in(v["s1_id"].unique().implode())).join(s1cnt, on=["country", "name_cat"]) \
       .select(s1_id="entity_id", a_s1_cat_df="s1_cat_df", country="country")
b = pl.concat([pl.scan_parquet(CACHE + f"norm_n4_train_s{i}.parquet").select(cand_id="entity_id", country="country", name_cat="name_cat")
               .filter(pl.col("cand_id").is_in(v["cand_id"].unique().implode())).collect() for i in (2, 3)])
b = b.join(s1cnt, on=["country", "name_cat"], how="left").select("cand_id", b_s1_cat_df=pl.col("s1_cat_df").fill_null(0))
v = v.join(a.drop("country"), on="s1_id", how="left").join(b, on="cand_id", how="left").with_columns(
    n_channels=pl.sum_horizontal([(pl.col(r) < 999).cast(pl.UInt8) for r in
                                  ("rank_n", "rank_a", "rank_c", "rrank_n", "rrank_a", "rrank_c")]))
S1CAT = ["a_s1_cat_df", "b_s1_cat_df", "n_channels"]

# ---- E-REVM: the pool record's reverse competition (from the cached reverse ranks)
rev_path = sorted(glob.glob(CACHE + "reverse_*_train.parquet"))[-1]
rev = (pl.scan_parquet(rev_path).select("s1", "rid", "rscore")
       .filter(pl.col("rid").is_in(v["rid"].unique().implode())).collect())
top = rev.group_by("rid").agg(r_best=pl.col("rscore").max(), r_second=pl.col("rscore").top_k(2).min(),
                              r_n=pl.len())
v = v.join(top, on="rid", how="left").with_columns(
    r_gap_best=pl.col("r_best") - pl.col("rscore"),                               # 0 if this S1 is the record's best
    r_margin12=pl.col("r_best") - pl.col("r_second"),                             # how clear the record's best S1 is
    r_ratio=pl.col("rscore") / pl.col("r_best"))
REVM = ["r_best", "r_gap_best", "r_margin12", "r_ratio", "r_n"]
print(f"prepared {len(v):,} pairs in {time.time()-t0:.0f}s", flush=True)

if run("E-MISS-X"):
    log({**evaluate(v, fit_predict(v, BASE_FEATURES + MX), ids, truth, "+miss_interactions"), "exp": "batch2"})
if run("E-MISS-SP"):
    base = fit_predict(v, BASE_FEATURES)
    e = v.with_row_index("_i").filter(pl.col("b_addr_empty"))
    sp = fit_predict(e.drop("_i"), BASE_FEATURES)                                # specialist trained on empty-address pairs
    mix = base.copy(); mix[e["_i"].to_numpy()] = sp
    log({**evaluate(v, mix, ids, truth, "+empty_addr_specialist"), "exp": "batch2"})
if run("E-REVM"):
    log({**evaluate(v, fit_predict(v, BASE_FEATURES + REVM), ids, truth, "+reverse_margins"), "exp": "batch2"})
if run("E-S1CAT"):
    log({**evaluate(v, fit_predict(v, BASE_FEATURES + S1CAT), ids, truth, "+s1cat_channels(rev63)"), "exp": "batch2"})
if run("E-HARDNEG"):
    v2 = v.with_columns(w=pl.when((pl.col("label") == 0) & pl.col("p1").is_between(0.3, 0.95)).then(3.0).otherwise(1.0))
    log({**evaluate(v2, fit_predict(v2, BASE_FEATURES, weight="w"), ids, truth, "+hardneg_weight3"), "exp": "batch2"})
if run("E-RANK"):
    rp = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [3, 5], "lambdarank_truncation_level": 20}
    log({**evaluate(v, fit_predict(v, BASE_FEATURES, params=rp, group_rank=True), ids, truth, "lambdarank"), "exp": "batch2"})
print(f"done {time.time()-t0:.0f}s")
