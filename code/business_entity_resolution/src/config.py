"""Experiment configuration.

Dataclass defaults, optionally overlaid by a JSON file, then by CLI flags.
Relative paths resolve against the project root (the folder that holds src/), so
the same config runs on a laptop, Kaggle or SageMaker. On SageMaker the standard
SM_CHANNEL_* / SM_MODEL_DIR env vars fill paths not given on the CLI.
"""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SplitCfg:
    seed: int = 42
    n_folds: int = 5
    val_fold: int = 0
    # Fraction of validation S1 used as blocking queries. Dev speed-up only:
    # the S2/S3 pool is never subsampled, so distractor density stays real.
    query_frac: float = 1.0


@dataclass
class BlockingCfg:
    key_types: list = field(default_factory=lambda: ["NT", "NP", "CAT", "AW", "HS", "NX"])
    df_cap: int = 5000        # keys shared by more pool records than this are dropped
    k_name: int = 10          # per (S1, source): top-k by name-key score
    k_addr: int = 10          #                   top-k by address-key score
    k_comb: int = 20          #                   top-k by combined score (union of the three)
    chunk: int = 20000        # S1 queries per join chunk (memory knob)
    # reverse blocking: every pool record queries an S1 index; its top-k S1s give (a) extra
    # candidates the S1-side ranking missed and (b) reverse-rank features for every pair
    reverse: bool = False
    k_rev_keep: int = 10      # reverse ranks kept (features); beyond -> 'not in top-k'
    k_rev_add: int = 5        # reverse pairs with best rank <= this are added as candidates
    # how equal scores are ordered inside a ranking: "rid" (legacy: by row position, which lets
    # records early in a file win every tie) or "hash" (pseudo-random but deterministic per pair)
    tie_break: str = "rid"


@dataclass
class ModelCfg:
    es_fold: int = 1          # early-stopping fold (never the validation fold)
    train_frac: float = 0.1   # fraction of train-fold S1 whose candidates become training pairs
    params: dict = field(default_factory=lambda: {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "seed": 42, "deterministic": True, "force_row_wise": True, "verbose": -1})
    num_boost_round: int = 3000
    early_stopping: int = 100
    stage2: bool = False      # second model on stage-1 features + per-S1 context of stage-1 scores
    extra_features: list = field(default_factory=list)   # opt-in features (features.EXTRA_FEATURES)
    transitive: bool = False  # stage 2 also gets one-hop support from the S1's confident matches
    stage1_es_from_train: bool = False   # early-stop stage 1 on a training-fold slice (not the stage-2 fold)
    es_frac: float = -1.0     # fraction of the es fold used for stage 2 (-1 = split.query_frac, legacy)
    drop_features: list = field(default_factory=list)    # base features to leave out (ablation-proven useless)


@dataclass
class DecisionCfg:
    exclusive: bool = True    # each S2/S3 record goes to at most its best-scoring S1


@dataclass
class Config:
    name: str = "base"
    data_dir: str = "../../student_resource/dataset"
    work_dir: str = "../../artifacts"
    split: SplitCfg = field(default_factory=SplitCfg)
    blocking: BlockingCfg = field(default_factory=BlockingCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)

    def path(self, attr):
        p = Path(getattr(self, attr))
        return p if p.is_absolute() else (ROOT / p).resolve()


def _merge(dc, d):
    for f in fields(dc):
        if f.name in d:
            cur = getattr(dc, f.name)
            if is_dataclass(cur):
                _merge(cur, d[f.name])
            else:
                setattr(dc, f.name, d[f.name])
    unknown = set(d) - {f.name for f in fields(dc)}
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")


def load(path=None, data=None):
    """Defaults, overlaid by a JSON file and/or an already-parsed dict (e.g. from a model bundle)."""
    cfg = Config()
    if path:
        _merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    if data:
        _merge(cfg, data)
    return cfg


def stage_hash(*parts):
    """Cache key for a pipeline stage: its config section(s) + code/library versions."""
    blob = json.dumps([asdict(p) if is_dataclass(p) else p for p in parts], sort_keys=True)
    blob += pl.__version__
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def cli(description, extra=None):
    """Shared CLI: --config, --data-dir, --work-dir. Returns (cfg, args)."""
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config")
    ap.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_DATA"))
    ap.add_argument("--work-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load(args.config)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.work_dir:
        cfg.work_dir = args.work_dir
    return cfg, args
