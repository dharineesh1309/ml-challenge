"""Offline comparison of decision rules on a train run's validation predictions.

    python decide_eval.py --run ../../../artifacts/runs/<ts>_dev_train

Rules: global threshold (sweep), + exclusivity, expected-F0.5 decoding (+ exclusivity).
Note: exclusivity here only sees competition among validation S1s (a subset), so its
measured gain is a lower bound of what it does on the full test S1 set.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

import config
import decision
import metrics
from data_io import read_ground_truth


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = config.load(data=json.loads((run / "bundle.json").read_text(encoding="utf-8"))["config"])
    pred = pl.read_parquet(run / "val_pairs.parquet", columns=["s1_id", "cand_id", "p"])
    pairs, n_true = read_ground_truth(cfg.path("data_dir"))
    # all validation S1 (those with no candidates too)
    from data_io import read_source
    from dataset import add_split
    s1 = add_split(read_source(cfg.path("data_dir"), "train", 1).select("entity_id"), cfg.split)
    ids = s1.filter(pl.col("is_query"))["entity_id"]
    truth = pairs.join(pl.DataFrame({"s1_id": ids}), on="s1_id")

    def score(matches):
        return metrics.summary(metrics.per_entity(ids, matches, truth))

    rows = []
    for name, pr in (("global", pred), ("global+excl", decision.exclusive(pred))):
        tab = decision.sweep(pr, ids, truth, np.round(np.arange(0.5, 0.96, 0.025), 3))
        t = decision.best_threshold(tab)
        rows.append({"rule": name, "param": t, **tab.filter(pl.col("t") == t).to_dicts()[0]})
    for excl in (False, True):
        pr = decision.exclusive(pred) if excl else pred
        for miss in (0.0, 0.1, 0.2):
            rows.append({"rule": f"expF{'+excl' if excl else ''}", "param": miss,
                         **score(decision.expected_f_decode(pr, missing=miss))})
    out = pl.DataFrame(rows).select("rule", "param", "macro_f05", "pair_precision", "pair_recall",
                                    "singleton_acc", "nonsingleton_empty_frac", "mean_pred_per_s1")
    out.write_csv(run / "decision_eval.tsv", separator="\t")
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print(out)


if __name__ == "__main__":
    main()
