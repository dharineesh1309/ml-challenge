"""E-ROBUST: which feature groups break cross-country transfer? (proxy for unseen France)
For each group G: train on country A without G, validate on country B (and in-country A->A).
A group worth dropping/normalizing hurts transfer but not in-country performance."""
import sys, time
sys.path.insert(0, "D:/ml-challenge/lab")
sys.path.insert(0, "D:/ml-challenge/code/business_entity_resolution/src")
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
import polars as pl
from harness import BASE_FEATURES, CACHE, PARAMS, evaluate, load, log

t0 = time.time()
v, ids, truth, _ = load()
c = (pl.scan_parquet(CACHE + "norm_n4_train_s1.parquet").select(s1_id="entity_id", country="country")
     .filter(pl.col("s1_id").is_in(v["s1_id"].unique().implode())).collect())
v = v.join(c, on="s1_id")
GROUPS = {
    "none": [],
    "idf_scores": ["score_n", "score_a", "score", "rscore", "nk"],
    "counts": ["n_cand_s1", "a_cat_df", "b_cat_df", "a_ntok", "b_ntok"],
    "abs_overlap": ["addr_w_inter", "num_inter", "sig_inter"],
    "script": ["a_nonlatin", "b_nonlatin", "b_addr_nonlatin"],
    "all_scale": ["score_n", "score_a", "score", "rscore", "nk", "n_cand_s1", "a_cat_df", "b_cat_df",
                  "a_ntok", "b_ntok", "addr_w_inter", "num_inter", "sig_inter"],
}


def run(train_c, val_c, feats):
    tr = v.filter(pl.col("half") & (pl.col("country") == train_c))
    te = v.filter(~pl.col("half") & (pl.col("country") == val_c))
    es = (tr["s1"] % 10 == 0).to_numpy()
    X = tr.select(feats).cast(pl.Float32).to_numpy(); y = tr["label"].to_numpy()
    b = lgb.train({**PARAMS, "seed": 0}, lgb.Dataset(X[~es], y[~es]), 3000,
                  valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(100, verbose=False)])
    p = b.predict(te.select(feats).cast(pl.Float32).to_numpy(), num_iteration=b.best_iteration)
    dst = te["s1_id"].unique()
    return evaluate(te, p, dst, truth.join(pl.DataFrame({"s1_id": dst}), on="s1_id"), "", slices=False)["macro_f05"]


for g, drop in GROUPS.items():
    feats = [f for f in BASE_FEATURES if f not in drop]
    r = {"name": f"robust -{g}", "US->US": run("US", "US", feats), "US->India": run("US", "India", feats),
         "India->India": run("India", "India", feats), "India->US": run("India", "US", feats),
         "n_feats": len(feats), "exp": "robust"}
    log(r)
print(f"done {time.time()-t0:.0f}s")
