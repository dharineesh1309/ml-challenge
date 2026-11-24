"""Batch 3: confirm E-REVM on a 2nd seed; test house + reverse margins combined (2 seeds).
Features computed with the PIPELINE's own functions (features.house_change_features), so the
lab result is what the main pipeline will produce."""
import glob, sys, time
sys.path.insert(0, "D:/ml-challenge/lab")
sys.stdout.reconfigure(encoding="utf-8")
import polars as pl
sys.path.insert(0, "D:/ml-challenge/code/business_entity_resolution/src")
from harness import BASE_FEATURES, CACHE, evaluate, fit_predict, load, log
from features import house_change_features, HOUSE_CHANGE, REVERSE_MARGINS

t0 = time.time()
v, ids, truth, _ = load()
a = (pl.scan_parquet(CACHE + "norm_n4_train_s1.parquet").select(s1_id="entity_id", ha="house_no")
     .filter(pl.col("s1_id").is_in(v["s1_id"].unique().implode())).collect())
b = pl.concat([pl.scan_parquet(CACHE + f"norm_n4_train_s{i}.parquet").select(cand_id="entity_id", hb="house_no")
               .filter(pl.col("cand_id").is_in(v["cand_id"].unique().implode())).collect() for i in (2, 3)])
v = v.join(a, on="s1_id", how="left").join(b, on="cand_id", how="left").with_columns(pl.col("ha", "hb").fill_null(""))
v = v.with_columns(**house_change_features(pl.col("ha"), pl.col("hb")))
rev = (pl.scan_parquet(sorted(glob.glob(CACHE + "reverse_*_train.parquet"))[-1]).select("rid", "rscore")
       .filter(pl.col("rid").is_in(v["rid"].unique().implode())).collect())
top = rev.group_by("rid").agg(r_best=pl.col("rscore").max(), r_second=pl.col("rscore").top_k(2).min(), r_n=pl.len())
v = v.join(top, on="rid", how="left").with_columns(
    r_gap_best=pl.col("r_best") - pl.col("rscore"), r_margin12=pl.col("r_best") - pl.col("r_second"),
    r_ratio=pl.col("rscore") / pl.col("r_best"))
print(f"prepared in {time.time()-t0:.0f}s", flush=True)
log({**evaluate(v, fit_predict(v, BASE_FEATURES + REVERSE_MARGINS, seed=1), ids, truth, "+reverse_margins_s1"), "exp": "batch3"})
for seed in (0, 1):
    log({**evaluate(v, fit_predict(v, BASE_FEATURES + HOUSE_CHANGE + REVERSE_MARGINS, seed=seed), ids, truth,
                    f"+house+reverse_margins_s{seed}"), "exp": "batch3"})
print(f"done {time.time()-t0:.0f}s")
