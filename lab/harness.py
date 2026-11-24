"""Isolated A/B harness for research ideas (does NOT touch the main pipeline or its artifacts).

Data: the saved validation pairs of the current best run (all features, labels), i.e. the
exact candidate set the main pipeline scores. S1 entities are split in two halves; a model is
trained on one half and scores the other, then swapped. Every variant uses the identical
protocol, so DIFFERENCES between variants are meaningful (absolute scores are lower than the
full model, which trains on ~9x more S1).

Reported: official macro F0.5 (exclusive + best global threshold on pooled out-of-fold scores),
pair P/R, loss split, and per-slice recall/FP.
"""
import glob
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

SRC = "D:/ml-challenge/code/business_entity_resolution/src"
sys.path.insert(0, SRC)
import config  # noqa: E402
import decision  # noqa: E402
import metrics  # noqa: E402
from data_io import read_ground_truth, read_source  # noqa: E402
from dataset import add_split  # noqa: E402

RUN = sorted(glob.glob("D:/ml-challenge/artifacts/runs/*_rev_tf30_s2_train"))[-1]
BUNDLE = json.load(open(RUN + "/bundle.json"))
BASE_FEATURES = BUNDLE["features"]
CACHE = "D:/ml-challenge/artifacts/cache/"
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 8}


def load():
    """Validation pairs + truth for ALL validation S1 (incl. those without candidates)."""
    cfg = config.load(data=BUNDLE["config"])
    v = pl.read_parquet(RUN + "/val_pairs.parquet")
    pairs, n_true = read_ground_truth(cfg.path("data_dir"))
    ids = add_split(read_source(cfg.path("data_dir"), "train", 1).select("entity_id"), cfg.split) \
        .filter(pl.col("is_query"))["entity_id"]
    truth = pairs.join(pl.DataFrame({"s1_id": ids}), on="s1_id")
    v = v.with_columns(half=(pl.col("s1") * 2654435761 % 1000 < 500))          # fixed S1 split
    return v, ids, truth, n_true


def fit_predict(v, feats, params=None, seed=0, weight=None, group_rank=False):
    """2-fold by S1 halves; returns out-of-fold scores aligned with v."""
    params = {**PARAMS, **(params or {}), "seed": seed}
    oof = np.zeros(len(v), dtype=np.float64)
    for h in (True, False):
        tr, te = v.filter(pl.col("half") == h), v.filter(pl.col("half") != h)
        es_mask = (tr["s1"] % 10 == 0).to_numpy()                                # 10% of train S1 for early stop
        X = tr.select(feats).cast(pl.Float32).to_numpy()
        y = tr["label"].to_numpy()
        w = tr[weight].to_numpy() if weight else None
        kw = {}
        if group_rank:                                                           # LambdaRank grouped by S1
            trs = tr.with_row_index("_i").sort("s1")
            order = trs["_i"].to_numpy()
            X, y = X[order], y[order]
            es_mask = es_mask[order]
            grp = lambda m: trs.filter(pl.Series(m)).group_by("s1", maintain_order=True).len()["len"].to_numpy()
            dtr = lgb.Dataset(X[~es_mask], y[~es_mask], group=grp(~es_mask))
            des = lgb.Dataset(X[es_mask], y[es_mask], group=grp(es_mask), reference=dtr)
        else:
            dtr = lgb.Dataset(X[~es_mask], y[~es_mask], weight=None if w is None else w[~es_mask])
            des = lgb.Dataset(X[es_mask], y[es_mask], reference=dtr)
        b = lgb.train(params, dtr, 3000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(100, verbose=False)], **kw)
        idx = np.where((v["half"] != h).to_numpy())[0]
        oof[idx] = b.predict(te.select(feats).cast(pl.Float32).to_numpy(), num_iteration=b.best_iteration)
    return oof


def evaluate(v, oof, ids, truth, name, slices=True):
    """Official macro F0.5 with the main pipeline's decision (exclusive + swept global threshold)."""
    pred = v.select("s1_id", "cand_id", "label", p=pl.Series(oof))
    is_rank = pred["p"].min() < 0 or pred["p"].max() > 1
    if is_rank:                                                                   # ranker scores -> percentile
        pred = pred.with_columns(p=pl.col("p").rank() / len(pred))
    ex = decision.exclusive(pred)
    ts = (np.unique(np.round(np.quantile(ex["p"].to_numpy(), np.linspace(0.90, 0.995, 80)), 6)) if is_rank
          else np.round(np.arange(0.3, 0.97, 0.01), 3))
    tab = decision.sweep(ex, ids, truth, ts)
    t = decision.best_threshold(tab)
    best = tab.filter(pl.col("t") == t).to_dicts()[0]
    m = decision.apply_threshold(ex, t).join(pred.select("s1_id", "cand_id", "label"), on=["s1_id", "cand_id"])
    f = lambda mm: metrics.per_entity(ids, mm, truth)["f05"].mean()
    nofp = f(m.filter(pl.col("label") == 1))
    oracle = f(pred.filter(pl.col("label") == 1))
    r = {"name": name, "macro_f05": round(best["macro_f05"], 5), "t": t,
         "P": round(best["pair_precision"], 4), "R": round(best["pair_recall"], 4),
         "singleton": round(best["singleton_acc"], 4),
         "loss_fp": round(nofp - best["macro_f05"], 4), "loss_clf": round(oracle - nofp, 4)}
    if slices:
        vv = v.with_columns(pred=pred["p"] >= t)
        sl = {}
        for col, expr in (("addr_empty", pl.col("b_addr_empty")),
                          ("house_diff", pl.col("house_both") & ~pl.col("house_eq"))):
            s = vv.filter(expr)
            sl[col] = {"FN": int(((s["label"] == 1) & ~s["pred"]).sum()), "FP": int(((s["label"] == 0) & s["pred"]).sum())}
        r["slices"] = sl
    return r


def log(r, path="D:/ml-challenge/lab/results.jsonl"):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n")
    print(json.dumps(r), flush=True)
