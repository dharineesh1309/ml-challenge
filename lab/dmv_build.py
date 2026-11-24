"""Density-matched validation (DMV).

Test has ~40% unmatched pool records vs ~26% in train. To reproduce test-like competition for the
validation S1s, HIDE a random 25% of the NON-validation S1 entities (=20% of all S1): their true
matches stay in the pool as orphaned look-alikes (26% -> ~41% unmatched). Validation S1 entities and
their ground truth are untouched, so no label information moves between folds.

Everything that depends on the S1 set is rebuilt for the reduced set: reverse ranks (separate cache
name, the baseline cache is not touched), reverse competition stats, S1-side name commonness.
Forward blocking depends only on the pool and is unchanged.

Outputs (lab/dmv/):
  pairs_normal.parquet  validation pairs at normal density, CURRENT feature code (base + extras + context)
  pairs_dmv.parquet     the same at test-like density
  baseline_eval.json    the frozen baseline model scored on both
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "D:/ml-challenge/code/business_entity_resolution/src")
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
import numpy as np
import polars as pl
import config, decision, metrics
from blocking import build_index, candidates, load_split, pool_keys, reverse_ranks, reverse_stats
from data_io import read_ground_truth
from dataset import add_split
from features import add_name_commonness, pair_features, s1_context, transitive_support

OUT = Path("D:/ml-challenge/lab/dmv"); OUT.mkdir(parents=True, exist_ok=True)
BASE = Path("D:/ml-challenge/artifacts/baseline_best")
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

cfg = config.load("D:/ml-challenge/code/business_entity_resolution/configs/rev63_hrt.json")  # same blocking as baseline
bundle = json.load(open(BASE / "bundle.json"))
b1 = lgb.Booster(model_file=str(BASE / "model.txt")); b2 = lgb.Booster(model_file=str(BASE / "model2.txt"))
f1, f2 = bundle["features"], bundle["stage2"]["features"]

s1_all, pool = load_split(cfg, "train")
s1_all = add_split(s1_all, cfg.split)
pkeys = pool_keys(cfg, "train", pool)
idx = build_index(pkeys, pool, cfg)
del pkeys
pairs, _ = read_ground_truth(cfg.path("data_dir"))
log(f"loaded: S1 {len(s1_all):,} pool {len(pool):,}")

hide_h = (pl.col("entity_id").str.slice(3).cast(pl.UInt64) * 2246822519 + 7) % 1000     # independent of split hash
REGIMES = {"normal": pl.lit(True),
           "dmv": (pl.col("fold") == cfg.split.val_fold) | (hide_h >= 250)}


def build(name, keep):
    s1 = s1_all.filter(keep)
    s1, pl_pool = add_name_commonness(s1, pool)
    rev = reverse_ranks(cfg, "train" if name == "normal" else "train_dmv20", s1, pl_pool)
    rs = reverse_stats(rev)
    q = s1.filter(pl.col("is_query"))
    cand = candidates(cfg, q, idx, (rev, rs))
    f = pair_features(cand, q, pl_pool)
    f = f.join(q.select("s1", s1_id="entity_id"), on="s1").join(pl_pool.select("rid", cand_id="entity_id"), on="rid")
    lab = pairs.join(q.select(s1_id="entity_id"), on="s1_id").with_columns(label=pl.lit(1, pl.UInt8))
    f = f.join(lab, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
    f = f.with_columns(p1=pl.Series(b1.predict(f.select(f1).cast(pl.Float32).to_numpy()), dtype=pl.Float32))
    f = transitive_support(s1_context(f), pl_pool)
    f = f.with_columns(p_base=pl.Series(b2.predict(f.select(f2).cast(pl.Float32).to_numpy()), dtype=pl.Float32))
    f.write_parquet(OUT / f"pairs_{name}.parquet")
    stats = {"S1_in_universe": len(s1), "pool": len(pool),
             "unmatched_pool_share": round(1 - pairs.join(s1.select(s1_id="entity_id"), on="s1_id").height / len(pool), 4),
             "val_S1": len(q), "cand_pairs": len(f), "cand_per_S1": round(len(f) / len(q), 1),
             "cand_p95": float(f.group_by("s1").len()["len"].quantile(0.95)),
             "rev_added_share": round(f["rev_added"].mean(), 4),
             "blocking_recall": round(int(f["label"].sum()) / lab.height, 4)}
    log(f"{name}: {stats}")
    return q["entity_id"], lab.select("s1_id", "cand_id"), f, stats


def score(ids, truth, f, col):
    p = f.select("s1_id", "cand_id", p=col)
    ex = decision.exclusive(p)
    tab = decision.sweep(ex, ids, truth, np.round(np.arange(0.5, 0.97, 0.01), 3))
    t = decision.best_threshold(tab); best = tab.filter(pl.col("t") == t).to_dicts()[0]
    at80 = tab.filter(pl.col("t") == 0.8).to_dicts()[0]
    oracle = metrics.summary(metrics.per_entity(ids, p.filter(f["label"] == 1), truth))["macro_f05"]
    keep = ["macro_f05", "pair_precision", "pair_recall", "fp", "fn", "singleton_acc", "mean_pred_per_s1"]
    return {"best_t": t, "at_best_t": {k: round(best[k], 4) for k in keep},
            "at_t0.80": {k: round(at80[k], 4) for k in keep}, "oracle": round(oracle, 4)}


res = {}
for name, keep in REGIMES.items():
    ids, truth, f, stats = build(name, keep)
    res[name] = {"stats": stats, "stage1": score(ids, truth, f, "p1"), "stage2_baseline": score(ids, truth, f, "p_base"),
                 "p_distribution": {q: round(float(f["p_base"].quantile(q)), 4) for q in (0.9, 0.95, 0.99)}}
    log(f"{name}: {json.dumps(res[name])}")
    del f
json.dump(res, open(OUT / "baseline_eval.json", "w"), indent=2)
log("done")
