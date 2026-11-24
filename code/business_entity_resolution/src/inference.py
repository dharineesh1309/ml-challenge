"""Test inference entry point: model bundle -> blocking -> features -> P(match) -> decision
-> output/matching_results.tsv + output/candidate_pairs.tsv -> official validator.

    python inference.py --model-dir M [--data-dir D] [--work-dir W] [--out-dir O] [--validator V]

All modelling settings (normalization, blocking budgets, features, threshold) come from the
bundle written by train.py; only paths come from the CLI / SageMaker env.
"""
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

import config
import decision
import submission
from blocking import candidates, prepare
from data_io import read_source
from features import TRANSITIVE, pair_features, s1_context, transitive_support
from normalize import VERSION


def extra_args(ap):
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR"), required=not os.environ.get("SM_MODEL_DIR"))
    ap.add_argument("--out-dir")
    ap.add_argument("--validator", help="utils/validate_submission.py (default: next to the dataset)")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--s1-chunk", type=int, default=100_000)
    ap.add_argument("--max-s1", type=int, default=0, help="DRY RUN: only the first N S1 (no validator)")
    ap.add_argument("--scores", help="reuse a saved pair_scores_<split>.parquet (skip blocking+model)")
    ap.add_argument("--threshold", type=float, help="override the bundle threshold")


def main():
    cli_cfg, args = config.cli("Run test inference and write the submission", extra_args)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    model_dir = Path(args.model_dir)
    bundle = json.loads((model_dir / "bundle.json").read_text(encoding="utf-8"))
    if bundle["version"]["normalize"] != VERSION and not args.scores:
        raise SystemExit(f"bundle normalize={bundle['version']['normalize']} but code is {VERSION}: retrain")
    cfg = config.load(data=bundle["config"])
    cfg.data_dir, cfg.work_dir = cli_cfg.data_dir, cli_cfg.work_dir
    feats = bundle["features"]
    t = args.threshold if args.threshold is not None else bundle["threshold"]
    out = Path(args.out_dir or os.environ.get("SM_OUTPUT_DATA_DIR") or cfg.path("work_dir").parent / "output")

    s1_ids = read_source(cfg.path("data_dir"), args.split, 1)["entity_id"]      # file order
    if args.max_s1:
        s1_ids = s1_ids.head(args.max_s1)
    if args.scores:
        pred = pl.read_parquet(args.scores)
        log(f"loaded scores: {len(pred):,} pairs")
    else:
        booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        st2 = bundle.get("stage2")
        booster2 = lgb.Booster(model_file=str(model_dir / "model2.txt")) if st2 else None
        s1, pool, idx, _, rev = prepare(cfg, args.split)
        if args.max_s1:
            s1 = s1.head(args.max_s1)            # dry run: same code path, fewer queries
        log(f"prepared {args.split}: S1 {len(s1):,}  pool {len(pool):,}  threshold {t}  stage2 {bool(st2)}")
        X = lambda d, fs: d.select(fs).cast(pl.Float32).to_numpy()
        preds = []
        for i in range(0, len(s1), args.s1_chunk):     # chunks by S1: all candidates of an S1 stay together
            q = s1.slice(i, args.s1_chunk)
            f = pair_features(candidates(cfg, q, idx, rev), q, pool)
            p = pl.Series(booster.predict(X(f, feats)), dtype=pl.Float32)
            if booster2:
                f = s1_context(f.with_columns(p1=p))
                if TRANSITIVE[0] in st2["features"]:
                    f = transitive_support(f, pool)
                p = pl.Series(booster2.predict(X(f, st2["features"])), dtype=pl.Float32)
            preds.append(f.select("s1", "rid", p=p))
            log(f"S1 {min(i + args.s1_chunk, len(s1)):,}/{len(s1):,}  pairs so far {sum(map(len, preds)):,}")
        pred = (pl.concat(preds).join(s1.select("s1", s1_id="entity_id"), on="s1")
                .join(pool.select("rid", cand_id="entity_id"), on="rid"))
        out.mkdir(parents=True, exist_ok=True)
        pred.write_parquet(out / f"pair_scores_{args.split}.parquet")    # for re-deciding without re-scoring

    cand_pairs = pred.select("s1_id", "cand_id", score="p")
    kept = decision.exclusive(pred) if cfg.decision.exclusive else pred
    match_pairs = decision.apply_threshold(kept, t)
    submission.write_outputs(out, s1_ids, cand_pairs, match_pairs)
    per = match_pairs.group_by("s1_id").len()
    log(f"wrote {out}: {len(cand_pairs):,} candidates, {len(match_pairs):,} matches (t={t}, "
        f"exclusive={cfg.decision.exclusive}), {len(s1_ids) - len(per):,} of {len(s1_ids):,} S1 with no match")

    validator = Path(args.validator) if args.validator else cfg.path("data_dir").parent / "utils" / "validate_submission.py"
    if args.split == "test" and validator.exists() and not args.max_s1:
        rc = submission.validate(out, cfg.path("data_dir") / "test", validator)
        log("validator PASS" if rc == 0 else "validator FAILED")
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
