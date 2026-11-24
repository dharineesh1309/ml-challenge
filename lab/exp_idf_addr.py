"""E-IDF: rarity-weighted address evidence (OOD / France hypothesis).

Observation: on test, 3.5% of accepted French pairs have two different rare street words (US 1.5%,
India 1.3%). French addresses carry many generic tokens (rue/des/de/la + multi-word regions) that
inflate UNWEIGHTED address similarity. Country-agnostic, label-free fix: weight address tokens by
their rarity in the SAME country's pool (IDF, computed transductively per split).

New features (pair):
  addr_idf_jacc     IDF-weighted Jaccard of address words
  addr_idf_shared   sum IDF of shared words
  addr_conflict     min(max IDF of words only in A, max IDF of words only in B)  (two different rare words)
  addr_unshared_a   max IDF of S1-address words missing from the candidate
  addr_unshared_b   max IDF of candidate-address words missing from the S1 address
Evaluated: in-domain (normal + DMV validation, 2 seeds) and US->India transfer proxy.
"""
import json, sys
sys.path.insert(0, "D:/ml-challenge/lab")
sys.path.insert(0, "D:/ml-challenge/code/business_entity_resolution/src")
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
import numpy as np
import polars as pl
from harness import BASE_FEATURES, PARAMS, evaluate, fit_predict, log
from data_io import read_ground_truth
import config
from features import HOUSE_CHANGE, REVERSE_MARGINS, S1_CONTEXT, TRANSITIVE

C = "D:/ml-challenge/artifacts/cache/"
D = "D:/ml-challenge/lab/dmv/"
cfg = config.load()
pairs, _ = read_ground_truth(cfg.path("data_dir"))
pool = pl.concat([pl.read_parquet(C + f"norm_n4_train_s{i}.parquet", columns=["entity_id", "country", "addr_words"]) for i in (2, 3)])
s1 = pl.read_parquet(C + "norm_n4_train_s1.parquet", columns=["entity_id", "country", "addr_words"])
n_c = pool.group_by("country").agg(n=pl.len())
idf = (pool.select("entity_id", "country", t="addr_words").explode("t").drop_nulls().unique(["entity_id", "t"])
       .group_by("country", "t").agg(df=pl.len()).join(n_c, on="country")
       .select("country", "t", idf=(pl.col("n") / pl.col("df")).log().cast(pl.Float32)))
print("idf table", len(idf), flush=True)


def add_idf_feats(v):
    x = (v.select("s1", "rid", "s1_id", "cand_id")
         .join(s1.select(s1_id="entity_id", country="country", a="addr_words"), on="s1_id")
         .join(pool.select(cand_id="entity_id", b="addr_words"), on="cand_id").with_row_index("i"))
    ta = x.select("i", "country", t="a").explode("t").drop_nulls().unique(["i", "t"]).with_columns(side=pl.lit("a"))
    tb = x.select("i", "country", t="b").explode("t").drop_nulls().unique(["i", "t"]).with_columns(side=pl.lit("b"))
    t = pl.concat([ta, tb]).join(idf, on=["country", "t"], how="left").with_columns(pl.col("idf").fill_null(0))
    t = t.with_columns(n_sides=pl.col("side").n_unique().over("i", "t"))
    g = t.group_by("i").agg(
        shared=pl.col("idf").filter(pl.col("n_sides") == 2).sum() / 2,
        union=pl.col("idf").filter(pl.col("n_sides") == 2).sum() / 2 + pl.col("idf").filter(pl.col("n_sides") == 1).sum(),
        ua=pl.col("idf").filter((pl.col("n_sides") == 1) & (pl.col("side") == "a")).max(),
        ub=pl.col("idf").filter((pl.col("n_sides") == 1) & (pl.col("side") == "b")).max())
    x = x.join(g, on="i", how="left").with_columns(
        addr_idf_shared=pl.col("shared"),
        addr_idf_jacc=pl.when(pl.col("union") > 0).then(pl.col("shared") / pl.col("union")),
        addr_unshared_a=pl.col("ua"), addr_unshared_b=pl.col("ub"),
        addr_conflict=pl.min_horizontal(pl.col("ua").fill_null(0), pl.col("ub").fill_null(0)))
    empty = (x["a"].list.len() == 0) | (x["b"].list.len() == 0)
    feats = x.select("s1", "rid", *IDF).with_columns([pl.when(pl.Series(empty)).then(None).otherwise(pl.col(c)).alias(c) for c in IDF])
    return v.join(feats, on=["s1", "rid"], how="left").with_columns(
        half=(pl.col("s1") * 2654435761 % 1000 < 500),
        country=pl.col("s1_id").replace_strict(dict(zip(s1["entity_id"], s1["country"])), default=None))


IDF = ["addr_idf_jacc", "addr_idf_shared", "addr_conflict", "addr_unshared_a", "addr_unshared_b"]
FULL = BASE_FEATURES + S1_CONTEXT + ["a_s1_cat_df", "b_s1_cat_df", "n_channels"] + HOUSE_CHANGE + REVERSE_MARGINS + TRANSITIVE
res = {}
for regime in ("normal", "dmv"):
    v = add_idf_feats(pl.read_parquet(D + f"pairs_{regime}.parquet"))
    ids = v["s1_id"].unique(); truth = pairs.join(pl.DataFrame({"s1_id": ids}), on="s1_id")
    for seed in (0, 1):
        log({**evaluate(v, fit_predict(v, FULL, seed=seed), ids, truth, f"{regime}:full_s{seed}"), "exp": "idf"})
        log({**evaluate(v, fit_predict(v, FULL + IDF, seed=seed), ids, truth, f"{regime}:full+idf_s{seed}"), "exp": "idf"})
    if regime == "dmv":   # transfer proxy for an unseen country: train on US half, evaluate India other half
        for feats, nm in ((FULL, "full"), (FULL + IDF, "full+idf")):
            for src_c, dst_c in (("US", "India"), ("India", "US")):
                tr = v.filter(pl.col("half") & (pl.col("country") == src_c)); te = v.filter(~pl.col("half") & (pl.col("country") == dst_c))
                es = (tr["s1"] % 10 == 0).to_numpy(); X = tr.select(feats).cast(pl.Float32).to_numpy(); y = tr["label"].to_numpy()
                b = lgb.train({**PARAMS, "seed": 0}, lgb.Dataset(X[~es], y[~es]), 3000, valid_sets=[lgb.Dataset(X[es], y[es])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                p = b.predict(te.select(feats).cast(pl.Float32).to_numpy(), num_iteration=b.best_iteration)
                dst = te["s1_id"].unique()
                r = evaluate(te, p, dst, pairs.join(pl.DataFrame({"s1_id": dst}), on="s1_id"), "", slices=False)
                log({"name": f"transfer {src_c}->{dst_c} {nm}", "macro_f05": r["macro_f05"], "P": r["P"], "R": r["R"],
                     "singleton": r["singleton"], "exp": "idf-transfer"})
print("done")
