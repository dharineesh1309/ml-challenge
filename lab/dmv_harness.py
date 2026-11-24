"""Validation matrix on NORMAL vs DENSITY-MATCHED (test-like) validation pairs.

Same 44,736 validation S1 entities and ground truth in both regimes; pairs built by the current
pipeline code (lab/dmv_build.py). Same 2-fold S1-half protocol as the lab harness, seeds 0 and 1.
p1 = frozen baseline stage-1 score (out-of-sample for validation S1).

M1  stack ablations evaluated at test-like density (control / +rev63 / +house / +revm / +trans / full)
M2  density-matched TRAINING: model trained on normal-density half vs DMV half, both scored on the DMV half
M3  threshold transfer: best threshold normal vs DMV; split-half threshold selection bias
M4  stage-2 data-size sensitivity (fraction of fitting S1 used)
"""
import json, sys
sys.path.insert(0, "D:/ml-challenge/lab")
sys.path.insert(0, "D:/ml-challenge/code/business_entity_resolution/src")
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
import numpy as np
import polars as pl
import decision, metrics
from harness import BASE_FEATURES, PARAMS, evaluate, fit_predict, log
from data_io import read_ground_truth
import config
from features import EXTRA_FEATURES, HOUSE_CHANGE, REVERSE_MARGINS, S1_CONTEXT, TRANSITIVE

D = "D:/ml-challenge/lab/dmv/"
cfg = config.load()
pairs, _ = read_ground_truth(cfg.path("data_dir"))


def load(name):
    v = pl.read_parquet(D + f"pairs_{name}.parquet").with_columns(half=(pl.col("s1") * 2654435761 % 1000 < 500))
    return v


vn, vd = load("normal"), load("dmv")
ids = vn["s1_id"].unique()                                  # every val S1 with candidates (same in both)
truth = pairs.join(pl.DataFrame({"s1_id": ids}), on="s1_id")
ctl = BASE_FEATURES + S1_CONTEXT                            # = frozen baseline's stage-2 feature set
REV63 = ["a_s1_cat_df", "b_s1_cat_df", "n_channels"]
stacks = {"control": ctl, "+rev63": ctl + REV63, "+house": ctl + HOUSE_CHANGE, "+revm": ctl + REVERSE_MARGINS,
          "+trans": ctl + TRANSITIVE, "full": ctl + REV63 + HOUSE_CHANGE + REVERSE_MARGINS + TRANSITIVE,
          "full-rev63": ctl + HOUSE_CHANGE + REVERSE_MARGINS + TRANSITIVE}
which = sys.argv[1:] or ["M1", "M2", "M3", "M4"]

if "M1" in which:
    for regime, v in (("dmv", vd), ("normal", vn)):
        for name, feats in stacks.items():
            if regime == "normal" and name not in ("control", "full", "full-rev63"):
                continue
            for seed in (0, 1):
                log({**evaluate(v, fit_predict(v, feats, seed=seed), ids, truth, f"{regime}:{name}_s{seed}"), "exp": "dmv-M1"})

if "M2" in which:
    full = stacks["full"]
    for seed in (0, 1):
        oof = np.zeros(len(vd))
        for h in (True, False):                               # train on NORMAL half h, score DMV other half
            tr = vn.filter(pl.col("half") == h); es = (tr["s1"] % 10 == 0).to_numpy()
            X = tr.select(full).cast(pl.Float32).to_numpy(); y = tr["label"].to_numpy()
            b = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X[~es], y[~es]), 3000,
                          valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(100, verbose=False)])
            te_mask = (vd["half"] != h).to_numpy()
            oof[te_mask] = b.predict(vd.filter(pl.col("half") != h).select(full).cast(pl.Float32).to_numpy(),
                                     num_iteration=b.best_iteration)
        log({**evaluate(vd, oof, ids, truth, f"train=normal -> eval=dmv full_s{seed}"), "exp": "dmv-M2"})
    # (train=dmv -> eval=dmv is M1 'dmv:full')

if "M3" in which:
    for regime, v in (("normal", vn), ("dmv", vd)):
        oof = fit_predict(v, stacks["full"], seed=0)
        p = v.select("s1_id", "cand_id", "half", p=pl.Series(oof))
        res = {}
        for pick, apply in ((True, False), (False, True)):       # choose t on one half, score the other
            ids_p = p.filter(pl.col("half") == pick)["s1_id"].unique(); ids_a = p.filter(pl.col("half") == apply)["s1_id"].unique()
            tp = decision.sweep(decision.exclusive(p.filter(pl.col("half") == pick).drop("half")), ids_p,
                                truth.join(pl.DataFrame({"s1_id": ids_p}), on="s1_id"), np.round(np.arange(0.5, 0.97, 0.01), 3))
            t = decision.best_threshold(tp)
            ta = decision.sweep(decision.exclusive(p.filter(pl.col("half") == apply).drop("half")), ids_a,
                                truth.join(pl.DataFrame({"s1_id": ids_a}), on="s1_id"), np.round(np.arange(0.5, 0.97, 0.01), 3))
            res[f"pick_half{int(pick)}"] = {"t_picked": t, "f05_at_picked_t": round(ta.filter(pl.col("t") == t)["macro_f05"].item(), 5),
                                           "f05_at_own_best_t": round(ta["macro_f05"].max(), 5),
                                           "own_best_t": float(ta.filter(pl.col("macro_f05") == ta["macro_f05"].max())["t"][0])}
        log({"name": f"M3 threshold transfer {regime}", **res, "exp": "dmv-M3"})

if "M4" in which:
    for frac in (0.25, 0.5, 1.0):
        v = vd.with_columns(keep=(pl.col("s1") * 40503 % 1000) < int(frac * 1000))
        oof = np.zeros(len(v)); full = stacks["full"]
        for h in (True, False):
            tr = v.filter((pl.col("half") == h) & pl.col("keep")); es = (tr["s1"] % 10 == 0).to_numpy()
            X = tr.select(full).cast(pl.Float32).to_numpy(); y = tr["label"].to_numpy()
            b = lgb.train({**PARAMS, "seed": 0}, lgb.Dataset(X[~es], y[~es]), 3000,
                          valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(100, verbose=False)])
            m_ = (v["half"] != h).to_numpy()
            oof[m_] = b.predict(v.filter(pl.col("half") != h).select(full).cast(pl.Float32).to_numpy(), num_iteration=b.best_iteration)
        log({**evaluate(v, oof, ids, truth, f"M4 stage2 fit on {int(frac * 100)}% of half-S1 (dmv)", slices=False), "exp": "dmv-M4"})
print("done")
