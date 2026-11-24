"""Official challenge metric: F0.5 per Source-1 entity, macro-averaged over ALL evaluated S1.

Per S1:  no true & no pred -> 1.0 ; no true & some pred -> 0.0 ; some true & no pred -> 0.0
         else F0.5 = 1.25*P*R / (0.25*P + R)  (0 when no true positive)
"""
import polars as pl


def per_entity(s1_ids: pl.Series, pred: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """s1_ids: evaluated S1 ids; pred/truth: long (s1_id, cand_id). Returns per-S1 counts and f05."""
    base = pl.DataFrame({"s1_id": s1_ids.unique()})
    n_pred = pred.group_by("s1_id").agg(n_pred=pl.len())
    n_true = truth.group_by("s1_id").agg(n_true=pl.len())
    tp = pred.join(truth, on=["s1_id", "cand_id"]).group_by("s1_id").agg(tp=pl.len())
    d = (base.join(n_pred, on="s1_id", how="left").join(n_true, on="s1_id", how="left")
         .join(tp, on="s1_id", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
         .when((pl.col("n_true") == 0) | (pl.col("n_pred") == 0) | (pl.col("tp") == 0)).then(0.0)
         .otherwise(1.25 * p * r / (0.25 * p + r)))
    return d.with_columns(f05=f)


def summary(d: pl.DataFrame) -> dict:
    single = d.filter(pl.col("n_true") == 0)
    tp, npred, ntrue = d["tp"].sum(), d["n_pred"].sum(), d["n_true"].sum()
    return {
        "macro_f05": d["f05"].mean(),
        "pair_precision": tp / npred if npred else 0.0,
        "pair_recall": tp / ntrue if ntrue else 0.0,
        "fp": int(npred - tp), "fn": int(ntrue - tp),
        "singleton_acc": single["f05"].mean() if len(single) else None,
        "nonsingleton_empty_frac": d.filter(pl.col("n_true") > 0).select((pl.col("n_pred") == 0).mean()).item(),
        "mean_pred_per_s1": d["n_pred"].mean(),
    }


if __name__ == "__main__":
    # the worked example from the problem statement, plus the singleton rules
    truth = pl.DataFrame({"s1_id": ["A", "A"], "cand_id": ["S2-47", "S3-812"]})
    pred = pl.DataFrame({"s1_id": ["A", "A", "A", "C"], "cand_id": ["S2-47", "S2-193", "S3-812", "S2-9"]})
    d = per_entity(pl.Series(["A", "B", "C"]), pred, truth).sort("s1_id")
    f = d["f05"].to_list()
    assert abs(f[0] - 0.714) < 1e-3, f   # statement example
    assert f[1] == 1.0 and f[2] == 0.0     # singleton: empty -> 1, any match -> 0
    print("metrics self-check OK", summary(d))
