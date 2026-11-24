"""Train/validation split by S1 entity (never by pair).

Each S2/S3 record belongs to at most one S1 (measured on train GT), so splitting
S1 entities also splits positives cleanly. The fold is a fixed integer hash of
the numeric ID part, so it is identical across runs, machines and library versions.
"""
import polars as pl


def _mix(seed):
    n = pl.col("entity_id").str.slice(3).cast(pl.UInt64)
    return (n * 2654435761 + seed * 40503) % (1 << 32)   # Knuth multiplicative hash


def add_split(s1: pl.DataFrame, sc) -> pl.DataFrame:
    """Adds fold (0..n_folds-1), bucket (0..999, for subsampling), in_frac (query_frac sample)
    and is_query (val fold & in_frac)."""
    h = _mix(sc.seed)
    return s1.with_columns(
        fold=((h // 256) % sc.n_folds).cast(pl.UInt8),
        bucket=(h % 1000).cast(pl.UInt16),
    ).with_columns(in_frac=pl.col("bucket") < int(sc.query_frac * 1000)).with_columns(
        is_query=(pl.col("fold") == sc.val_fold) & pl.col("in_frac"))
