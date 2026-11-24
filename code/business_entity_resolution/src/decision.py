"""Turning pair probabilities into per-S1 match sets.

Baseline: global threshold chosen by the official macro F0.5 on validation S1s
(all of them: singletons and S1s with no candidates included).
"""
import numpy as np
import polars as pl

import metrics


def apply_threshold(pred: pl.DataFrame, t: float) -> pl.DataFrame:
    """pred: (s1_id, cand_id, p) -> matches (s1_id, cand_id, score)."""
    return pred.filter(pl.col("p") >= t).select("s1_id", "cand_id", score="p")


def sweep(pred, s1_ids, truth, ts=None) -> pl.DataFrame:
    ts = ts if ts is not None else np.round(np.arange(0.05, 0.96, 0.025), 3)
    rows = []
    for t in ts:
        d = metrics.per_entity(s1_ids, apply_threshold(pred, t), truth)
        rows.append({"t": float(t), **metrics.summary(d)})
    return pl.DataFrame(rows)


def exclusive(pred: pl.DataFrame) -> pl.DataFrame:
    """Each S2/S3 record may belong to at most one S1 (0 overlaps in train GT):
    keep a pair only if this S1 is the candidate's highest-probability S1."""
    return pred.filter(pl.col("p") == pl.col("p").max().over("cand_id"))


def expected_f_decode(pred: pl.DataFrame, min_p: float = 0.05, missing: float = 0.0) -> pl.DataFrame:
    """Per-S1 subset choice maximizing expected F0.5 (Instacart / Jansche-style plug-in).

    Candidates sorted by p; for top-k:  E[F] ~= 1.25 * sum_{i<=k} p_i / (k + 0.25 * N),
    N = sum of all p + `missing` (expected true matches lost by blocking). The empty set
    scores P(no true match) = prod(1 - p_i) (only if missing == 0). Returns (s1_id, cand_id, score)."""
    d = (pred.filter(pl.col("p") >= min_p).sort(["s1_id", "p"], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("s1_id"),
                       cum=pl.col("p").cum_sum().over("s1_id"),
                       n=pl.col("p").sum().over("s1_id") + missing))
    d = d.with_columns(ef=1.25 * pl.col("cum") / (pl.col("k") + 0.25 * pl.col("n")))
    p_empty = (pred.group_by("s1_id").agg(e=(1 - pl.col("p")).log().sum().exp())
               .with_columns(pl.col("e") * (1.0 if missing == 0 else 0.0)))
    best = d.group_by("s1_id").agg(best_ef=pl.col("ef").max(),
                                   best_k=pl.col("k").filter(pl.col("ef") == pl.col("ef").max()).first())
    best = best.join(p_empty, on="s1_id", how="left").filter(pl.col("best_ef") > pl.col("e").fill_null(0))
    return (d.join(best.select("s1_id", "best_k"), on="s1_id").filter(pl.col("k") <= pl.col("best_k"))
            .select("s1_id", "cand_id", score="p"))


def best_threshold(table: pl.DataFrame) -> float:
    """Max macro F0.5; among near-ties (within 0.001) take the most conservative (highest) t,
    because the test pool is denser than train (5.76 vs 4.68 records per S1)."""
    top = table["macro_f05"].max()
    return float(table.filter(pl.col("macro_f05") >= top - 0.001)["t"].max())
