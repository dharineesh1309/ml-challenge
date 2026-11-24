"""Training entry point (same code on laptop, Kaggle and SageMaker).

blocking candidates -> pair features -> LightGBM -> macro-F0.5 threshold -> model bundle

    python train.py --config ../configs/dev.json [--data-dir D] [--work-dir W] [--model-dir M]

S1 roles (by entity, never by pair): 'train' = train folds (subsampled by model.train_frac),
'es' = early-stopping fold, 'val' = validation fold (threshold + reported metrics only).
Blocking always searches the FULL S2/S3 pool, exactly as at test time.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import polars as pl

import config
import decision
import metrics
from blocking import candidates, log_experiment, peak_gb, prepare
from data_io import read_ground_truth
from dataset import add_split
from features import FEATURES, S1_CONTEXT, TRANSITIVE, pair_features, s1_context, transitive_support
from normalize import VERSION


def labelled_pairs(cfg, s1, pool, idx, q, rev=None):
    """Candidates + features + label for query S1 rows q (which carry a 'role' column)."""
    cand = candidates(cfg, q, idx, rev)
    feats = pair_features(cand, q, pool)
    pairs, _ = read_ground_truth(cfg.path("data_dir"))
    lab = (pairs.join(q.select("s1", s1_id="entity_id"), on="s1_id")
           .join(pool.select("rid", cand_id="entity_id"), on="cand_id").select("s1", "rid", label=pl.lit(1, pl.UInt8)))
    return (feats.join(q.select("s1", "role", s1_id="entity_id"), on="s1")
            .join(pool.select("rid", cand_id="entity_id"), on="rid")
            .join(lab, on=["s1", "rid"], how="left").with_columns(pl.col("label").fill_null(0))), pairs


def fit(m, tr, es, feats):
    """LightGBM on tr with early stopping on es."""
    X = lambda d: d.select(feats).cast(pl.Float32).to_numpy()
    dtr = lgb.Dataset(X(tr), tr["label"].to_numpy(), feature_name=feats, free_raw_data=True)
    des = lgb.Dataset(X(es), es["label"].to_numpy(), reference=dtr)
    return lgb.train(m.params, dtr, m.num_boost_round, valid_sets=[des],
                     callbacks=[lgb.early_stopping(m.early_stopping, verbose=False), lgb.log_evaluation(200)])


def predict(booster, d, feats):
    return pl.Series(booster.predict(d.select(feats).cast(pl.Float32).to_numpy()), dtype=pl.Float32)


def main():
    cfg, args = config.cli("Train the pair matcher",
                           lambda ap: ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR")))
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    run = cfg.path("work_dir") / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.name}_train"
    run.mkdir(parents=True)
    model_dir = Path(args.model_dir) if args.model_dir else cfg.path("work_dir") / "models" / cfg.name
    model_dir.mkdir(parents=True, exist_ok=True)

    s1, pool, idx, _, rev = prepare(cfg, "train")
    s1 = add_split(s1, cfg.split)
    m, sp = cfg.model, cfg.split
    role = (pl.when(pl.col("is_query")).then(pl.lit("val"))
            .when((pl.col("fold") == m.es_fold)
                  & (pl.col("in_frac") if m.es_frac < 0 else pl.col("bucket") < int(m.es_frac * 1000)))
            .then(pl.lit("es"))
            .when((pl.col("fold") != sp.val_fold) & (pl.col("fold") != m.es_fold)
                  & (pl.col("bucket") < int(m.train_frac * 1000))).then(pl.lit("train")))
    q = s1.with_columns(role=role).filter(pl.col("role").is_not_null())
    log(f"prepared; query S1 by role: {dict(q['role'].value_counts().iter_rows())}")

    data, gt_pairs = labelled_pairs(cfg, s1, pool, idx, q, rev)
    del idx, rev
    counts = {r: {"pairs": int(n), "pos": int(p)} for r, n, p in
              data.group_by("role").agg(pl.len(), pl.col("label").sum()).iter_rows()}
    log(f"pairs: {counts}")

    part = lambda r: data.filter(pl.col("role") == r)
    tr, es, va = part("train"), part("es"), part("val")
    del data                                  # the three role slices are copies: free the combined table
    feats = FEATURES + [f for f in m.extra_features if f not in FEATURES]   # opt-in extras (config)
    feats = [f for f in feats if f not in m.drop_features]
    if m.stage1_es_from_train:
        # early-stop stage 1 on 10% of the TRAINING S1 (by S1 id), so the es fold that stage 2 learns
        # from is untouched by stage 1 in every way (fitting AND tree-count selection)
        es_mask = (pl.col("s1") * 2654435761 % 1000) < 100
        booster = fit(m, tr.filter(~es_mask), tr.filter(es_mask), feats)
    else:
        booster = fit(m, tr, es, feats)
    log(f"stage 1 trained: {booster.best_iteration} rounds")
    del tr                                    # stage-1 training pairs are not needed any more
    p_va = predict(booster, va, feats)
    stage2 = None
    if m.stage2:
        # stage 1 never saw the es fold, so its es scores are out-of-sample: stage 2 learns on
        # es (half fit / half early-stop by S1 parity); val stays untouched for the threshold.
        es2 = s1_context(es.with_columns(p1=predict(booster, es, feats)))
        va = s1_context(va.with_columns(p1=p_va))
        f2 = feats + S1_CONTEXT
        if m.transitive:                     # one-hop support from the S1's other confident matches
            es2, va = transitive_support(es2, pool), transitive_support(va, pool)
            f2 = f2 + TRANSITIVE
        booster2 = fit(m, es2.filter(pl.col("s1") % 2 == 0), es2.filter(pl.col("s1") % 2 == 1), f2)
        log(f"stage 2 trained: {booster2.best_iteration} rounds")
        p_va = predict(booster2, va, f2)
        booster2.save_model(str(model_dir / "model2.txt"), num_iteration=booster2.best_iteration)
        stage2 = {"features": f2, "best_iteration": booster2.best_iteration,
                  "feature_gain": dict(sorted(zip(f2, booster2.feature_importance("gain").round(1).tolist()),
                                              key=lambda x: -x[1])[:20])}
    elif (model_dir / "model2.txt").exists():
        (model_dir / "model2.txt").unlink()

    pred = va.select("s1_id", "cand_id", "label", p=p_va)
    val_ids = q.filter(pl.col("role") == "val")["entity_id"]
    truth = gt_pairs.join(pl.DataFrame({"s1_id": val_ids}), on="s1_id")
    # tune the threshold with the SAME decoder inference uses (exclusive best-S1 assignment first)
    decoded = decision.exclusive(pred) if cfg.decision.exclusive else pred
    table = decision.sweep(decoded, val_ids, truth)
    t = decision.best_threshold(table)
    best = table.filter(pl.col("t") == t).to_dicts()[0]
    oracle = metrics.summary(metrics.per_entity(val_ids, pred.filter(pl.col("label") == 1), truth))
    log(f"val macro F0.5 = {best['macro_f05']:.4f} at t={t}  (oracle within candidates {oracle['macro_f05']:.4f})")

    booster.save_model(str(model_dir / "model.txt"), num_iteration=booster.best_iteration)
    imp = dict(sorted(zip(feats, booster.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1]))
    bundle = {"version": {"normalize": VERSION, "polars": pl.__version__, "lightgbm": lgb.__version__},
              "config": config.asdict(cfg), "features": feats, "threshold": t,
              "val": best, "oracle_val": oracle, "pair_counts": counts, "best_iteration": booster.best_iteration,
              "stage2": stage2,
              "feature_gain": imp, "runtime_s": round(time.time() - t0), "peak_mem_gb": peak_gb()}
    (model_dir / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    table.write_csv(run / "threshold_sweep.tsv", separator="\t")
    va.with_columns(p=p_va).write_parquet(run / "val_pairs.parquet")    # for error analysis
    (run / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(table.select("t", "macro_f05", "pair_precision", "pair_recall", "singleton_acc",
                       "nonsingleton_empty_frac", "mean_pred_per_s1"))
    log_experiment(cfg, {"time": f"{datetime.now():%m-%d %H:%M}", "experiment": cfg.name, "stage": "train",
                         "train_pairs": counts["train"]["pairs"], "precision": round(best["pair_precision"], 4),
                         "recall": round(best["pair_recall"], 4), "macro_f05": round(best["macro_f05"], 4),
                         "singleton_acc": round(best["singleton_acc"] or -1, 4), "threshold": t,
                         "runtime_s": bundle["runtime_s"], "peak_mem_gb": bundle["peak_mem_gb"],
                         "notes": f"oracle={oracle['macro_f05']:.4f} iters={booster.best_iteration} "
                                  f"stage2={bool(stage2)} norm={VERSION}"})
    log(f"bundle: {model_dir}   run: {run}")


if __name__ == "__main__":
    main()
