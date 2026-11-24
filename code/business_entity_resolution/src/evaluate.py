"""Error analysis for a train run: dumps the most confident mistakes with raw + normalized
records and the key features, so fixes are chosen from evidence.

    python evaluate.py --run ../../../artifacts/runs/<ts>_dev_train [--t 0.5] [--n 400]

Writes errors_{fp,fn,singleton_fp,tp_lowconf}.tsv and slices.tsv into the run folder.
Blocking misses come from blocking.py's blocking_misses.tsv.
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

import config
from data_io import read_ground_truth
from normalize import normalized

SHOW = ["name_tset", "sig_jw", "skel_tset", "addr_tset", "house_eq", "house_contain", "num_jacc",
        "cat_eq", "a_cat_df", "b_cat_df", "score_rel", "rank_c", "b_addr_empty", "b_nonlatin"]
REC = ["entity_id", "name_raw", "addr_raw", "name_sig", "addr_norm"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--t", type=float)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    run = Path(a.run)
    bundle = json.loads((run / "bundle.json").read_text(encoding="utf-8"))
    cfg = config.load(data=bundle["config"])
    t = a.t if a.t is not None else bundle["threshold"]

    va = pl.read_parquet(run / "val_pairs.parquet")
    s1 = normalized(cfg, "train", 1).select([pl.col(c).alias("s1_" + c) for c in REC] + ["country"])
    pool = pl.concat([normalized(cfg, "train", k) for k in (2, 3)]).select([pl.col(c).alias("c_" + c) for c in REC])
    _, n_true = read_ground_truth(cfg.path("data_dir"))
    va = (va.join(n_true.rename({"s1_id": "s1_entity_id"}), left_on="s1_id", right_on="s1_entity_id", how="left")
          .with_columns(pred=pl.col("p") >= t))
    rows = (va.join(s1, left_on="s1_id", right_on="s1_entity_id").join(pool, left_on="cand_id", right_on="c_entity_id"))
    cols = ["p", "label", "n_true", "country", "src", *["s1_" + c for c in REC[1:]], *["c_" + c for c in REC[1:]], *SHOW]
    sets = {
        "fp": rows.filter(pl.col("pred") & (pl.col("label") == 0)).sort("p", descending=True),
        "singleton_fp": rows.filter(pl.col("pred") & (pl.col("n_true") == 0)).sort("p", descending=True),
        "fn": rows.filter(~pl.col("pred") & (pl.col("label") == 1)).sort("p"),
        "tp_lowconf": rows.filter(pl.col("pred") & (pl.col("label") == 1)).sort("p"),
    }
    for name, d in sets.items():
        d.select(cols).head(a.n).write_csv(run / f"errors_{name}.tsv", separator="\t")
        print(f"{name:14s} {len(d):8,d}")

    # pair-level precision/recall by slice at threshold t
    sl = []
    for col in ("country", "src", "b_addr_empty", "b_nonlatin", "house_eq", "cat_eq"):
        g = rows.group_by(col).agg(
            pairs=pl.len(), pos=pl.col("label").sum(),
            precision=(pl.col("pred") & (pl.col("label") == 1)).sum() / pl.col("pred").sum(),
            recall=(pl.col("pred") & (pl.col("label") == 1)).sum() / pl.col("label").sum())
        sl.append(g.with_columns(slice=pl.lit(col), value=pl.col(col).cast(pl.Utf8)).drop(col))
    sl = pl.concat(sl).select("slice", "value", "pairs", "pos", "precision", "recall")
    sl.write_csv(run / "slices.tsv", separator="\t")
    with pl.Config(tbl_rows=40):
        print(sl)


if __name__ == "__main__":
    main()
