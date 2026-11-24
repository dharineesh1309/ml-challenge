"""Candidate generation: IDF-weighted inverted-index top-K (Sparkly-style), plus a recall report.

Every record emits typed keys (all implicitly scoped to its country):
  NT   name signature token  (reordered names, typos in other tokens)
  NS   name token skeleton   (vowel typos, anyascii spellings of Indic scripts)
  NP   name token pair       (names made only of common words)
  CAT  concatenated name     (domains / hashtags: maurewilliamscolombier.com)
  AW   address word          (reordered / abbreviated addresses)
  HS   house no x addr word  (most selective address signal)
  NX   name skeleton token x addr word ('mdrn|jaipur': common name + common city is a rare
       combination; also bridges native-script names). Counts in both channels.
Keys held by more than `df_cap` pool records are dropped. Shared keys add their
IDF to a name score and/or an address score. Candidates per (S1, source) are the
union of the top k_name by name score, top k_addr by address score and top k_comb
by their sum, so empty-address and native-script matches are not crowded out.

    python blocking.py --config ../configs/dev.json [--split test]
"""
import json
import math
import time
from datetime import datetime

import polars as pl
import psutil

import config
from data_io import cached, read_ground_truth
from dataset import add_split
from normalize import VERSION, normalized

KT_BITS = {"NT": 1, "AW": 2, "HS": 4, "CAT": 8, "NS": 16, "NP": 32, "NX": 64}
KT_NAMES = {v: k for k, v in KT_BITS.items()}
NAME_BITS = [1, 8, 16, 32, 64]
ADDR_BITS = [2, 4, 64]


def _name_pairs(rec, id_col, n=5):
    """Order-free pairs of the first n signature tokens: 'life|baptist'."""
    t = pl.col("name_sig").str.split(" ")
    parts = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = t.list.get(i, null_on_oob=True), t.list.get(j, null_on_oob=True)
            pair = pl.concat_str(pl.min_horizontal(a, b), pl.lit("|"), pl.max_horizontal(a, b))
            parts.append(rec.select(id_col, "cc", tok=pl.when(a.is_not_null() & b.is_not_null()).then(pair)))
    return pl.concat(parts).drop_nulls("tok")


def build_keys(rec: pl.DataFrame, key_types, id_col="rid") -> pl.DataFrame:
    """rec needs id_col, cc, name_sig, name_skel, name_cat, addr_words, house_no -> (id, key u64, kt u8)."""
    min2 = pl.col("tok").str.len_chars() >= 2
    parts = []
    if "NT" in key_types:
        parts.append(rec.select(id_col, "cc", tok=pl.col("name_sig").str.split(" "), kt=pl.lit(1, pl.UInt8))
                     .explode("tok").filter(min2))
    if "NS" in key_types:
        parts.append(rec.select(id_col, "cc", tok=pl.col("name_skel").str.split(" "), kt=pl.lit(16, pl.UInt8))
                     .explode("tok").filter(min2))
    if "NP" in key_types:
        parts.append(_name_pairs(rec, id_col).with_columns(kt=pl.lit(32, pl.UInt8)))
    if "AW" in key_types:
        parts.append(rec.select(id_col, "cc", tok="addr_words", kt=pl.lit(2, pl.UInt8)).explode("tok").filter(min2))
    if "HS" in key_types:
        parts.append(rec.filter(pl.col("house_no") != "")
                     .select(id_col, "cc", "house_no", tok="addr_words").explode("tok").filter(min2)
                     .select(id_col, "cc", tok=pl.concat_str("house_no", pl.lit("|"), "tok"), kt=pl.lit(4, pl.UInt8)))
    if "CAT" in key_types:
        parts.append(rec.select(id_col, "cc", tok="name_cat", kt=pl.lit(8, pl.UInt8))
                     .filter(pl.col("tok").str.len_chars() >= 4))
    if "NX" in key_types:
        x = (rec.select(id_col, "cc", n=pl.col("name_skel").str.split(" ").list.head(3), a="addr_words")
             .explode("n").filter(pl.col("n").str.len_chars() >= 2)
             .explode("a").filter(pl.col("a").str.len_chars() >= 3))
        parts.append(x.select(id_col, "cc", tok=pl.concat_str("n", pl.lit("|"), "a"), kt=pl.lit(64, pl.UInt8)))
    k = pl.concat(parts)
    key = (pl.col("tok").hash(0) ^ (pl.col("kt").cast(pl.UInt64) * pl.lit(1 << 56, pl.UInt64))
           ^ (pl.col("cc").cast(pl.UInt64) * pl.lit(1 << 48, pl.UInt64)))
    return k.select(id_col, key=key, kt="kt").unique([id_col, "key"])


def load_split(cfg, split):
    """Normalized S1 and S2+S3 pool with integer ids and an open-set country code."""
    s1 = normalized(cfg, split, 1)
    pool = pl.concat([normalized(cfg, split, 2), normalized(cfg, split, 3)])
    countries = pl.Enum(sorted(set(s1["country"].unique()) | set(pool["country"].unique())))
    cc = pl.col("country").cast(countries).to_physical().cast(pl.UInt8)
    s1 = s1.with_row_index("s1").with_columns(cc=cc)
    pool = pool.with_row_index("rid").with_columns(cc=cc)
    return s1, pool


def pool_keys(cfg, split, pool):
    """All pool keys with their document frequency (pre-cap). Cached: the expensive part."""
    b = cfg.blocking
    path = cfg.path("work_dir") / "cache" / f"poolkeys_{config.stage_hash(VERSION, sorted(b.key_types))}_{split}.parquet"

    def build():
        step = 1_000_000   # chunked: the exploded key strings are the memory peak
        parts = []
        for i in range(0, len(pool), step):
            sl = pool.slice(i, step)
            parts.append(build_keys(sl, b.key_types).join(sl.select("rid", "src", "cc"), on="rid"))
        k = pl.concat(parts)
        del parts
        return k.join(k.group_by("key").agg(df=pl.len().cast(pl.UInt32)), on="key")
    return cached(path, build)


def build_index(pkeys, pool, cfg):
    n_c = pool.group_by("cc").agg(n=pl.len())
    return (pkeys.filter(pl.col("df") <= cfg.blocking.df_cap).join(n_c, on="cc")
            .select("key", "rid", "src", "kt", idf=(pl.col("n") / pl.col("df")).log().cast(pl.Float32)))


def _rank(col):
    """1-based rank within (s1, src), best first; ties broken by rid (frame is pre-sorted); 0-score -> no rank."""
    r = pl.col(col).rank("ordinal", descending=True).over("s1", "src").cast(pl.UInt16)
    return pl.when(pl.col(col) > 0).then(r).otherwise(None)


def query(qkeys, idx, cfg, kmax=None):
    """qkeys (s1, key, kt) -> ranked candidates (s1, rid, src, score_n, score_a, score, nk, ktmask,
    rank_n, rank_a, rank_c), keeping pairs within kmax on any channel (default: the config budgets)."""
    b = cfg.blocking
    ids = qkeys["s1"].unique().sort()
    out = []
    idf = pl.col("idf")
    for i in range(0, len(ids), b.chunk):
        qc = qkeys.filter(pl.col("s1").is_in(ids[i:i + b.chunk].implode()))
        g = (qc.drop("kt").join(idx, on="key")
             .group_by("s1", "rid").agg(src=pl.col("src").first(),
                                        score_n=idf.filter(pl.col("kt").is_in(NAME_BITS)).sum(),
                                        score_a=idf.filter(pl.col("kt").is_in(ADDR_BITS)).sum(),
                                        nk=pl.len().cast(pl.UInt16), ktmask=pl.col("kt").bitwise_or()))
        g = g.with_columns(score=pl.col("score_n") + pl.col("score_a"))
        if b.tie_break == "hash":      # ties broken per pair, not by file position
            g = g.with_columns(_tb=pl.struct("s1", "rid").hash(7)).sort("s1", "src", "_tb").drop("_tb")
        else:
            g = g.sort("s1", "src", "rid")
        g = g.with_columns(rank_n=_rank("score_n"), rank_a=_rank("score_a"), rank_c=_rank("score"))
        out.append(g.filter(keep(kmax or b.k_name, kmax or b.k_addr, kmax or b.k_comb)))
    return pl.concat(out)


def prepare(cfg, split, keep_pool_keys=False):
    """Everything query-independent for a split: normalized S1 + pool (with name commonness),
    the capped blocking index and (optionally) the reverse ranks. Shared by the report,
    train.py and inference.py."""
    from features import add_name_commonness
    s1, pool = load_split(cfg, split)
    s1, pool = add_name_commonness(s1, pool)
    pkeys = pool_keys(cfg, split, pool)
    idx = build_index(pkeys, pool, cfg)
    if not keep_pool_keys:
        del pkeys
        pkeys = None
    rev = reverse_ranks(cfg, split, s1, pool) if cfg.blocking.reverse else None
    if rev is not None:
        rev = (rev, reverse_stats(rev))          # (reverse ranks, per-record competition stats)
    return s1, pool, idx, pkeys, rev


def reverse_ranks(cfg, split, s1, pool):
    """For EVERY pool record: its top-k_rev_keep S1s per channel, from an index over S1 keys.
    Returns (s1, rid, rrank_n, rrank_a, rrank_c, rscore). Cached per split (~1 h to build)."""
    b = cfg.blocking
    parts = [VERSION, sorted(b.key_types), b.df_cap, b.k_rev_keep] + ([b.tie_break] if b.tie_break != "rid" else [])
    tag = config.stage_hash(*parts)       # legacy tie-break keeps the existing cache key
    path = cfg.path("work_dir") / "cache" / f"reverse_{tag}_{split}.parquet"

    def build():
        t0 = time.time()
        rec = s1.rename({"s1": "rid"})
        k = build_keys(rec, b.key_types).join(rec.select("rid", "cc"), on="rid")
        k = k.with_columns(src=pl.lit(1, pl.UInt8)).join(k.group_by("key").agg(df=pl.len()), on="key")
        n_c = s1.group_by("cc").agg(n=pl.len())
        s1_idx = (k.filter(pl.col("df") <= b.df_cap).join(n_c, on="cc")
                  .select("key", "rid", "src", "kt", idf=(pl.col("n") / pl.col("df")).log().cast(pl.Float32)))
        del k
        out, step = [], 500_000
        for i in range(0, len(pool), step):
            q = pool.slice(i, step).rename({"rid": "s1"})          # pool records act as queries
            r = query(build_keys(q, b.key_types, id_col="s1"), s1_idx, cfg, kmax=b.k_rev_keep)
            out.append(r.select(s1="rid", rid="s1", rrank_n="rank_n", rrank_a="rank_a", rrank_c="rank_c",
                                rscore="score").join(q.select(rid="s1", src="src"), on="rid"))
            print(f"  reverse {min(i + step, len(pool)):,}/{len(pool):,}  {time.time() - t0:.0f}s", flush=True)
        return pl.concat(out)
    return cached(path, build)


def reverse_stats(rev):
    """Per pool record: its best and 2nd-best reverse score over ALL S1 and how many S1 it kept.
    Must be computed on the full reverse table (not a query subset)."""
    return rev.group_by("rid").agg(r_best=pl.col("rscore").max(), r_second=pl.col("rscore").top_k(2).min(),
                                   r_n=pl.len().cast(pl.UInt16))


def candidates(cfg, q, idx, rev=None, rstats=None):
    """Final candidate set for query S1 rows q (the set written to candidate_pairs.tsv):
    forward top-k, plus (if rev) pairs where q's S1 is in a pool record's reverse top-k_rev_add,
    with reverse-rank columns attached to every pair."""
    fwd = query(build_keys(q, cfg.blocking.key_types, id_col="s1"), idx, cfg)
    if isinstance(rev, tuple):                   # prepare() returns (reverse ranks, reverse stats)
        rev, rstats = rev
    if rev is None:
        return fwd
    r = rev.join(q.select("s1"), on="s1", how="semi")
    best = pl.min_horizontal(pl.col("rrank_n", "rrank_a", "rrank_c").fill_null(999))
    add = (r.filter(best <= cfg.blocking.k_rev_add).select("s1", "rid", "src")
           .join(fwd.select("s1", "rid"), on=["s1", "rid"], how="anti"))
    c = pl.concat([fwd.with_columns(rev_added=pl.lit(False)), add.with_columns(rev_added=pl.lit(True))],
                  how="diagonal_relaxed")
    c = c.join(r.drop("src"), on=["s1", "rid"], how="left")
    if rstats is not None:
        c = c.join(rstats, on="rid", how="left")
    return c


def keep(kn, ka, kc):
    return ((pl.col("rank_n") <= kn).fill_null(False) | (pl.col("rank_a") <= ka).fill_null(False)
            | (pl.col("rank_c") <= kc).fill_null(False))


# ---------------------------------------------------------------- report

def _q(s, p):
    return float(s.quantile(p, "nearest")) if len(s) else 0.0


def candidate_stats(q, cand, pool):
    per = q.select("s1", "cc").join(cand.group_by("s1").len(), on="s1", how="left").fill_null(0)
    pool_n = pool.group_by("cc").agg(n=pl.len())
    all_pairs = q.join(pool_n, on="cc")["n"].sum()
    n = per["len"]
    return {"n_query_s1": len(q), "cand_total": int(n.sum()), "cand_mean": round(n.mean(), 2),
            "cand_median": _q(n, 0.5), "cand_p95": _q(n, 0.95), "cand_max": int(n.max()),
            "s1_with_zero_cand": int((n == 0).sum()), "reduction_ratio": 1 - n.sum() / all_pairs}


GRID_MAX = 50


def budget_grid(ranked, truth, n_query):
    """Recall vs mean candidates/S1 for every (k_name, k_addr, k_comb) budget; Pareto rows only."""
    t = truth.join(ranked.select("s1", "rid", "rank_n", "rank_a", "rank_c"), on=["s1", "rid"], how="left")
    rows = []
    for kn in (0, 3, 5, 10, 20, 30):
        for ka in (0, 3, 5, 10, 20, 30):
            for kc in (0, 5, 10, 20, 30, 50):
                rows.append({"k_name": kn, "k_addr": ka, "k_comb": kc,
                             "cand_mean": ranked.select(keep(kn, ka, kc).sum()).item() / n_query,
                             "recall": t.select(keep(kn, ka, kc).mean()).item()})
    g = pl.DataFrame(rows).sort("cand_mean")
    g = g.filter(pl.col("recall") > pl.col("recall").cum_max().shift(1).fill_null(-1))   # Pareto frontier
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in g.iter_rows(named=True)]


def recall_report(cfg, q, qkeys, pool, pkeys, cand, truth):
    """truth: (s1, rid, src, cc, country, cand_nonlatin, cand_addr_empty, n_true)."""
    t = truth.join(cand.select("s1", "rid", "ktmask", rank="rank_c"), on=["s1", "rid"], how="left")
    t = t.with_columns(rank=pl.when(pl.col("ktmask").is_not_null()).then(pl.col("rank").fill_null(0)))
    rep = {"true_pairs": len(t), "recall": t["rank"].is_not_null().mean()}

    # ceiling before top-K: does the true pair share ANY key (uncapped / within cap)?
    tk = pkeys.join(t.select("rid").unique(), on="rid")
    shared = qkeys.drop("kt").join(tk, on="key").join(t.select("s1", "rid"), on=["s1", "rid"], how="semi")
    any_key = shared.select("s1", "rid").unique().with_columns(any_key=pl.lit(True))
    in_cap = (shared.filter(pl.col("df") <= cfg.blocking.df_cap).select("s1", "rid").unique()
              .with_columns(in_cap=pl.lit(True)))
    t = (t.join(any_key, on=["s1", "rid"], how="left").join(in_cap, on=["s1", "rid"], how="left")
         .with_columns(pl.col("any_key", "in_cap").fill_null(False)))
    rep["ceiling_any_shared_key"] = t["any_key"].mean()
    rep["ceiling_within_df_cap"] = t["in_cap"].mean()

    found = pl.col("rank").is_not_null()
    slices = {}
    for col in ("country", "src", "cand_nonlatin", "cand_addr_empty", "n_true_bucket"):
        s = t.group_by(col).agg(n=pl.len(), recall=found.mean()).sort(col)
        slices[col] = {str(r[col]): {"n": r["n"], "recall": round(r["recall"], 4)} for r in s.iter_rows(named=True)}
    rep["recall_by"] = slices

    kt = {}
    for name, bit in KT_BITS.items():
        if name not in cfg.blocking.key_types:
            continue
        has = (pl.col("ktmask") & bit) > 0
        kt[name] = {"cand_pairs_with_key": int(cand.select(has.sum()).item()),
                    "true_found_with_key": int(t.filter(found).select(has.sum()).item()),
                    "true_found_only_by_key": int(t.filter(pl.col("ktmask") == bit).height)}
    rep["key_types"] = kt
    rep["miss_reason"] = {
        "no_shared_key": int(t.filter(~found & ~pl.col("any_key")).height),
        "only_keys_above_df_cap": int(t.filter(~found & pl.col("any_key") & ~pl.col("in_cap")).height),
        "below_top_k": int(t.filter(~found & pl.col("in_cap")).height),
    }
    return rep, t.filter(~found)


def miss_table(misses, q, pool, n=3000):
    cols = ["entity_id", "name_raw", "addr_raw", "name_core", "addr_norm", "house_no"]
    m = misses.head(n)
    m = m.join(q.select("s1", *[pl.col(c).alias("s1_" + c) for c in cols]), on="s1")
    m = m.join(pool.select("rid", *[pl.col(c).alias("c_" + c) for c in cols]), on="rid")
    reason = (pl.when(~pl.col("any_key")).then(pl.lit("no_shared_key"))
              .when(~pl.col("in_cap")).then(pl.lit("above_df_cap")).otherwise(pl.lit("below_top_k")))
    return m.with_columns(reason=reason).select(
        "reason", "country", "src", *["s1_" + c for c in cols], *["c_" + c for c in cols])


def log_experiment(cfg, row):
    path = cfg.path("work_dir") / "experiments.tsv"
    cols = ["time", "experiment", "stage", "blocking_recall", "cand_mean", "cand_p95", "train_pairs",
            "precision", "recall", "macro_f05", "singleton_acc", "threshold", "runtime_s", "peak_mem_gb", "notes"]
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write("\t".join(cols) + "\n")
        f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")


def peak_gb():
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)


def main():
    cfg, args = config.cli("Blocking + recall report",
                           lambda ap: ap.add_argument("--split", default="train", choices=["train", "test"]))
    t0 = time.time()
    run = cfg.path("work_dir") / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.name}_blocking_{args.split}"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(config.asdict(cfg), indent=2), encoding="utf-8")

    s1, pool = load_split(cfg, args.split)
    print(f"[{time.time() - t0:6.0f}s] loaded: S1 {len(s1):,}  pool {len(pool):,}")
    s1 = add_split(s1, cfg.split)
    q = s1.filter(pl.col("is_query" if args.split == "train" else "in_frac"))
    pkeys = pool_keys(cfg, args.split, pool)
    print(f"[{time.time() - t0:6.0f}s] pool keys: {len(pkeys):,}")
    idx = build_index(pkeys, pool, cfg)
    qkeys = build_keys(q, cfg.blocking.key_types, id_col="s1")
    ranked = query(qkeys, idx, cfg, kmax=GRID_MAX)
    b = cfg.blocking
    cand = ranked.filter(keep(b.k_name, b.k_addr, b.k_comb))
    print(f"[{time.time() - t0:6.0f}s] candidates: {len(cand):,} for {len(q):,} S1")

    rep = {"split": args.split, "candidates": candidate_stats(q, cand, pool)}
    rep["cand_by_country"] = {r["country"]: r["mean"] for r in cand.group_by("s1").len()
                              .join(q.select("s1", "country"), on="s1", how="right").fill_null(0)
                              .group_by("country").agg(mean=pl.col("len").mean()).iter_rows(named=True)}
    rep["df_cap_drop_frac_by_kt"] = {KT_NAMES[r["kt"]]: round(r["frac"], 4) for r in pkeys.group_by("kt")
                                     .agg(frac=(pl.col("df") > cfg.blocking.df_cap).mean()).iter_rows(named=True)}
    if args.split == "train":
        pairs, n_true = read_ground_truth(cfg.path("data_dir"))
        bucket = (pl.when(pl.col("n_true") <= 2).then(pl.lit("1-2")).when(pl.col("n_true") <= 4)
                  .then(pl.lit("3-4")).otherwise(pl.lit("5+")))
        truth = (pairs.join(q.select("s1", "cc", "country", s1_id="entity_id"), on="s1_id")
                 .join(pool.select("rid", "src", cand_id="entity_id",
                                   cand_nonlatin="name_nonlatin", cand_addr_empty=pl.col("addr_raw") == ""),
                       on="cand_id")
                 .join(n_true, on="s1_id").with_columns(n_true_bucket=bucket))
        r, misses = recall_report(cfg, q, qkeys, pool, pkeys, cand, truth)
        rep.update(r)
        rep["budget_grid_pareto"] = budget_grid(ranked, truth, len(q))
        miss_table(misses, q, pool).write_csv(run / "blocking_misses.tsv", separator="\t")
    rep["runtime_s"] = round(time.time() - t0)
    rep["peak_mem_gb"] = peak_gb()
    (run / "blocking_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    c = rep["candidates"]
    log_experiment(cfg, {"time": f"{datetime.now():%m-%d %H:%M}", "experiment": cfg.name,
                         "stage": f"blocking_{args.split}", "blocking_recall": round(rep.get("recall", -1), 4),
                         "cand_mean": c["cand_mean"], "cand_p95": c["cand_p95"], "runtime_s": rep["runtime_s"],
                         "peak_mem_gb": rep["peak_mem_gb"],
                         "notes": f"k={b.k_name}/{b.k_addr}/{b.k_comb} cap={b.df_cap} {b.key_types} norm={VERSION}"})
    print(f"report: {run}")


if __name__ == "__main__":
    main()
