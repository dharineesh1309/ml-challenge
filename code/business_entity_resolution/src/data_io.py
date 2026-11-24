"""Raw TSV readers and a tiny parquet cache.

Named data_io (not io) to avoid shadowing the stdlib module.
All source columns are read as UTF-8 strings with quoting disabled, because
addresses contain quotes and commas; missing values become ''.
"""
from pathlib import Path

import polars as pl

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8")
    return df.with_columns(pl.all().fill_null(""))


def read_source(data_dir, split, src):
    df = read_tsv(Path(data_dir) / split / f"{split}_source{src}.tsv")
    assert df.columns == SOURCE_COLS, df.columns
    return df


def read_ground_truth(data_dir):
    """Long form: one row per true (s1_id, cand_id) pair, plus n_true per S1 (0 for singletons)."""
    gt = read_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    gt = gt.with_columns(ids=pl.col("matched_entity_ids").str.split(","))
    n_true = gt.select(s1_id="source1_entity_id",
                       n_true=pl.col("ids").list.eval(pl.element().filter(pl.element() != "")).list.len())
    pairs = (gt.select(s1_id="source1_entity_id", cand_id="ids").explode("cand_id")
             .filter(pl.col("cand_id") != ""))
    return pairs, n_true


def cached(path, build):
    """Return parquet at `path` if present, else build() it, save, and return."""
    path = Path(path)
    if path.exists():
        return pl.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df
