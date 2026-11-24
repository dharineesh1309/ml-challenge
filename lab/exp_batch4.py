"""Batch 4.
E-TRANS   (TransClean-style transitive consistency / one-hop S1 profile): compare each candidate with
          the S1's OTHER confident matches (stage-1 p1 >= 0.9; p1 is out-of-sample for validation).
          Stage-2-level evidence, so the control is BASE + S1_CONTEXT (stage-2 feature set).
E-XCOUNTRY diagnostic: train on one country, validate on the other (France-transfer risk).
"""
import sys, time
sys.path.insert(0, "D:/ml-challenge/lab")
sys.path.insert(0, "D:/ml-challenge/code/business_entity_resolution/src")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from harness import BASE_FEATURES, CACHE, evaluate, fit_predict, load, log
from features import S1_CONTEXT

t0 = time.time()
v, ids, truth, _ = load()
cols = ["entity_id", "name_core", "addr_norm", "house_no", "country"]
a = (pl.scan_parquet(CACHE + "norm_n4_train_s1.parquet").select(cols)
     .filter(pl.col("entity_id").is_in(v["s1_id"].unique().implode())).collect())
b = pl.concat([pl.scan_parquet(CACHE + f"norm_n4_train_s{i}.parquet").select(cols[:4])
               .filter(pl.col("entity_id").is_in(v["cand_id"].unique().implode())).collect() for i in (2, 3)])
v = v.join(a.select(s1_id="entity_id", country="country"), on="s1_id").join(
    b.rename({"entity_id": "cand_id", "name_core": "c_name", "addr_norm": "c_addr", "house_no": "c_house"}), on="cand_id")

# ---- one-hop support: every candidate x the S1's confident matches (seeds), excluding itself
SEED = 0.9
seeds = v.filter(pl.col("p1") >= SEED).select("s1", seed_rid="rid", seed_src="src", s_name="c_name",
                                               s_addr="c_addr", s_house="c_house")
x = (v.select("s1", "rid", "src", "c_name", "c_addr", "c_house").join(seeds, on="s1")
     .filter(pl.col("rid") != pl.col("seed_rid")))
print(f"pairs {len(v):,}  seed links {len(x):,}  ({time.time()-t0:.0f}s)", flush=True)
x = x.with_columns(
    sn=pl.Series(process.cpdist(x["c_name"].to_list(), x["s_name"].to_list(), scorer=fuzz.token_set_ratio, workers=8)),
    sa=pl.Series(process.cpdist(x["c_addr"].to_list(), x["s_addr"].to_list(), scorer=fuzz.token_set_ratio, workers=8)),
).with_columns(
    sa=pl.when((pl.col("c_addr") == "") | (pl.col("s_addr") == "")).then(None).otherwise(pl.col("sa")),
    sh=(pl.col("c_house") != "") & (pl.col("c_house") == pl.col("s_house")),
    xsrc=pl.col("src") != pl.col("seed_src"))
sup = x.group_by("s1", "rid").agg(
    t_nseed=pl.len(), t_name_max=pl.col("sn").max(), t_name_mean=pl.col("sn").mean(),
    t_addr_max=pl.col("sa").max(), t_house_any=pl.col("sh").any().cast(pl.Float32),
    t_xsrc_name_max=pl.col("sn").filter(pl.col("xsrc")).max(),
    t_name_exact_any=(pl.col("sn") == 100).any().cast(pl.Float32))
v = v.join(sup, on=["s1", "rid"], how="left").with_columns(
    pl.col("t_nseed").fill_null(0),
    t_name_vs_s1=pl.col("t_name_max") - pl.col("name_tset"))        # >0: closer to the S1's observed variants than to S1
TRANS = ["t_nseed", "t_name_max", "t_name_mean", "t_addr_max", "t_house_any", "t_xsrc_name_max",
         "t_name_exact_any", "t_name_vs_s1"]
print(f"prepared in {time.time()-t0:.0f}s", flush=True)

S2F = BASE_FEATURES + S1_CONTEXT
for seed in (0, 1):
    log({**evaluate(v, fit_predict(v, S2F, seed=seed), ids, truth, f"stage2ctl_s{seed}"), "exp": "batch4"})
for seed in (0, 1):
    log({**evaluate(v, fit_predict(v, S2F + TRANS, seed=seed), ids, truth, f"stage2+trans_s{seed}"), "exp": "batch4"})

# ---- E-XCOUNTRY: train on one country's S1 half, score the other country's S1 half
import harness, lightgbm as lgb
from harness import PARAMS
for src_c, dst_c in (("US", "India"), ("India", "US")):
    for train_c in (dst_c, src_c):                              # in-country reference first, then transfer
        tr = v.filter(pl.col("half") & (pl.col("country") == train_c))
        te = v.filter(~pl.col("half") & (pl.col("country") == dst_c))
        es = (tr["s1"] % 10 == 0).to_numpy()
        X = tr.select(BASE_FEATURES).cast(pl.Float32).to_numpy(); y = tr["label"].to_numpy()
        bst = lgb.train({**PARAMS, "seed": 0}, lgb.Dataset(X[~es], y[~es]), 3000,
                        valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(100, verbose=False)])
        p = bst.predict(te.select(BASE_FEATURES).cast(pl.Float32).to_numpy(), num_iteration=bst.best_iteration)
        te_ids = te["s1_id"].unique()
        dst_ids = pl.Series([i for i in ids.to_list()]).filter(
            pl.Series(ids).is_in(v.filter(~pl.col("half") & (pl.col("country") == dst_c))["s1_id"].unique().implode()))
        r = evaluate(te, p, dst_ids, truth.join(pl.DataFrame({"s1_id": dst_ids}), on="s1_id"),
                     f"xcountry train={train_c} -> val={dst_c}", slices=False)
        log({**r, "exp": "xcountry"})
print(f"done {time.time()-t0:.0f}s")
