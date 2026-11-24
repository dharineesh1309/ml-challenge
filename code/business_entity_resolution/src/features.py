"""Pair features for (S1, candidate) rows.

Vectorised: rapidfuzz.process.cpdist (element-wise, multithreaded) for string
similarities, polars list/set ops for token and number overlap, plus the blocking
scores and ranks as context. Country is never a feature (France is unseen in train).
Similarities involving an empty field are null, so LightGBM treats them as missing.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

REC_COLS = ["name_core", "name_sig", "name_skel", "name_cat", "addr_norm", "addr_words", "addr_nums",
            "house_no", "name_nonlatin", "addr_nonlatin", "cat_df", "s1_cat_df", "legal_form"]

# (feature, column, scorer) -- scorers return 0..100 (JaroWinkler 0..1)
STRING_SIMS = [
    ("name_ratio", "name_core", fuzz.ratio),
    ("name_tset", "name_core", fuzz.token_set_ratio),
    ("name_tsort", "name_core", fuzz.token_sort_ratio),
    ("name_partial", "name_core", fuzz.partial_ratio),
    ("sig_ratio", "name_sig", fuzz.ratio),
    ("sig_jw", "name_sig", JaroWinkler.normalized_similarity),
    ("skel_tset", "name_skel", fuzz.token_set_ratio),
    ("cat_partial", "name_cat", fuzz.partial_ratio),
    ("addr_ratio", "addr_norm", fuzz.ratio),
    ("addr_tset", "addr_norm", fuzz.token_set_ratio),
    ("addr_partial", "addr_norm", fuzz.partial_ratio),
]


def add_name_commonness(s1, pool):
    """cat_df = pool records in the same country with the identical name signature (Splink-style
    term frequency: agreeing on a rare name is strong evidence, on a common one weak).
    s1_cat_df = S1 records with that name: if only one S1 carries a name, an address-less pool
    record with that name most likely belongs to it."""
    counts = pool.group_by("cc", "name_cat").agg(cat_df=pl.len().cast(pl.UInt32))
    s1_counts = s1.group_by("cc", "name_cat").agg(s1_cat_df=pl.len().cast(pl.UInt32))
    add = lambda d: (d.join(counts, on=["cc", "name_cat"], how="left")
                     .join(s1_counts, on=["cc", "name_cat"], how="left")
                     .with_columns(pl.col("cat_df", "s1_cat_df").fill_null(0)))
    return add(s1), add(pool)


def _jacc(a, b):
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return inter, pl.when(union > 0).then(inter / union)


def house_change_features(ha: pl.Expr, hb: pl.Expr) -> dict:
    """How two different house numbers differ. On this data, true-match noise changes a random
    digit or drops one, while look-alike negatives sit at NEARBY numbers (e.g. 2405 vs 2404:
    only 5% of |diff| 3-10 pairs match, but 90% of one-digit-changed large jumps do).
    Edit distance cannot separate these; numeric distance + where the digits differ can.
    Null when either side is missing, equal, or longer than 9 digits."""
    ok = (ha != "") & (hb != "") & (ha != hb) & (ha.str.len_chars() <= 9) & (hb.str.len_chars() <= 9)
    a, b = ha.cast(pl.Int64, strict=False), hb.cast(pl.Int64, strict=False)
    same_len = ha.str.len_chars() == hb.str.len_chars()
    digit = lambda x, k: (x // (10 ** k)) % 10
    n = ha.str.len_chars()
    diffs = [(digit(a, k) != digit(b, k)) & (pl.lit(k) < n) for k in range(9)]       # k = 0 is the last digit
    ham = pl.sum_horizontal([d.cast(pl.Int32) for d in diffs])
    # highest differing position k (from the right) -> position from the left, 0 = leading digit
    hi = pl.max_horizontal([pl.when(d).then(pl.lit(k)) for k, d in enumerate(diffs)])
    first_pos = (n - 1 - hi) / n
    transposed = pl.all_horizontal([ha.str.count_matches(str(d)) == hb.str.count_matches(str(d)) for d in range(10)])
    w = lambda e: pl.when(ok).then(e)
    return {"house_absdiff": w((a - b).abs().cast(pl.Float32)),
            "house_reldiff": w(((a - b).abs() / pl.max_horizontal(a, b)).cast(pl.Float32)),
            "house_hamming": w(pl.when(same_len).then(ham.cast(pl.Float32))),
            "house_first_diff_pos": w(pl.when(same_len).then(first_pos.cast(pl.Float32))),
            "house_transposed": w(pl.when(same_len).then(transposed.cast(pl.Float32)).otherwise(0.0))}


HOUSE_CHANGE = ["house_absdiff", "house_reldiff", "house_hamming", "house_first_diff_pos", "house_transposed"]
# reverse competition: how this S1 compares with the OTHER S1s competing for the same pool record
REVERSE_MARGINS = ["r_best", "r_gap_best", "r_margin12", "r_ratio", "r_n"]
# opt-in extras (config model.extra_features); computed always, used only when listed
EXTRA_FEATURES = HOUSE_CHANGE + REVERSE_MARGINS


def _block(c: pl.DataFrame) -> pl.DataFrame:
    """Features for one chunk of joined rows (a_* = S1 side, b_* = candidate side)."""
    feats = {}
    for name, col, scorer in STRING_SIMS:
        a, b = c["a_" + col].to_list(), c["b_" + col].to_list()
        v = process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)
        feats[name] = v
    feats["house_ed"] = process.cpdist(c["a_house_no"].to_list(), c["b_house_no"].to_list(),
                                       scorer=Levenshtein.distance, workers=-1).astype(np.float32)
    out = c.with_columns(**{k: pl.Series(v) for k, v in feats.items()})
    empty = {"name": (pl.col("a_name_core") == "") | (pl.col("b_name_core") == ""),
             "addr": (pl.col("a_addr_norm") == "") | (pl.col("b_addr_norm") == "")}
    out = out.with_columns([pl.when(empty["addr" if col.startswith("addr") else "name"]).then(None)
                            .otherwise(pl.col(name)).alias(name) for name, col, _ in STRING_SIMS])

    w_inter, w_jacc = _jacc("a_addr_words", "b_addr_words")
    n_inter, n_jacc = _jacc("a_addr_nums", "b_addr_nums")
    ta, tb = pl.col("a_name_sig").str.split(" "), pl.col("b_name_sig").str.split(" ")
    t_inter = ta.list.set_intersection(tb).list.len()
    ha, hb = pl.col("a_house_no"), pl.col("b_house_no")
    both_h = (ha != "") & (hb != "")
    la, lb = pl.col("a_legal_form"), pl.col("b_legal_form")
    return out.with_columns(
        house_ed=pl.when(both_h).then(pl.col("house_ed")),
        house_prefix=both_h & (ha != hb) & (ha.str.starts_with(hb) | hb.str.starts_with(ha)),   # 2108 vs 210
        legal_eq=(la != "") & (la == lb),
        legal_conflict=(la != "") & (lb != "") & (la != lb),                                 # Pvt Ltd vs LLP
        legal_one_missing=(la == "") != (lb == ""),
        addr_w_inter=w_inter, addr_w_jacc=w_jacc, num_inter=n_inter, num_jacc=n_jacc,
        sig_inter=t_inter, sig_jacc=t_inter / ta.list.set_union(tb).list.len(),
        house_both=both_h, house_eq=both_h & (ha == hb),
        house_contain=both_h & (ha != hb) & (ha.str.ends_with(hb) | hb.str.ends_with(ha)),
        cat_eq=(pl.col("a_name_cat") == pl.col("b_name_cat")) & (pl.col("a_name_cat") != ""),
        cat_contain=pl.col("a_name_cat").str.contains(pl.col("b_name_cat"), literal=True)
        | pl.col("b_name_cat").str.contains(pl.col("a_name_cat"), literal=True),
        a_ntok=ta.list.len(), b_ntok=tb.list.len(),
        a_addr_empty=pl.col("a_addr_norm") == "", b_addr_empty=pl.col("b_addr_norm") == "",
        a_nonlatin=pl.col("a_name_nonlatin"), b_nonlatin=pl.col("b_name_nonlatin"),
        b_addr_nonlatin=pl.col("b_addr_nonlatin"),
        a_cat_df=pl.col("a_cat_df"), b_cat_df=pl.col("b_cat_df"),
        a_s1_cat_df=pl.col("a_s1_cat_df"), b_s1_cat_df=pl.col("b_s1_cat_df"),
        **house_change_features(ha, hb),
    )


CONTEXT = ["src", "score_n", "score_a", "score", "nk", "rank_n", "rank_a", "rank_c",
           "score_rel", "score_n_rel", "score_a_rel", "n_cand_s1",
           "kt_NT", "kt_AW", "kt_HS", "kt_CAT", "kt_NS", "kt_NP",
           # reverse blocking: rank of this S1 among the pool record's best S1s (999 = not in top-k)
           "rrank_n", "rrank_a", "rrank_c", "rrank_best", "rscore", "rev_added"]
REVERSE_COLS = {"rrank_n": pl.UInt16, "rrank_a": pl.UInt16, "rrank_c": pl.UInt16, "rscore": pl.Float32,
                "rev_added": pl.Boolean}
PAIR = [n for n, _, _ in STRING_SIMS] + [
    "addr_w_inter", "addr_w_jacc", "num_inter", "num_jacc", "sig_inter", "sig_jacc",
    "house_both", "house_eq", "house_contain", "house_prefix", "house_ed",
    "legal_eq", "legal_conflict", "legal_one_missing", "cat_eq", "cat_contain", "a_ntok", "b_ntok",
    "a_addr_empty", "b_addr_empty", "a_nonlatin", "b_nonlatin", "b_addr_nonlatin", "a_cat_df", "b_cat_df",
    "a_s1_cat_df", "b_s1_cat_df", "n_channels"]
FEATURES = CONTEXT + PAIR


S1_CONTEXT = ["p1", "p1_rank", "p1_rank_src", "p1_max", "p1_second", "p1_gap", "p1_sum", "p1_n50",
              "p1_max_src", "p1_n50_src"]


def s1_context(d: pl.DataFrame) -> pl.DataFrame:
    """Stage-2 features: where this pair's stage-1 probability p1 sits among all candidates of
    the same S1 (and of the same S1 x source). All candidates of an S1 must be in d."""
    p = pl.col("p1")
    return d.with_columns(
        p1_rank=p.rank("ordinal", descending=True).over("s1").cast(pl.UInt16),
        p1_rank_src=p.rank("ordinal", descending=True).over("s1", "src").cast(pl.UInt16),
        p1_max=p.max().over("s1"),
        p1_second=p.top_k(2).min().over("s1"),
        p1_sum=p.sum().over("s1"),
        p1_n50=(p >= 0.5).sum().over("s1").cast(pl.UInt16),
        p1_max_src=p.max().over("s1", "src"),
        p1_n50_src=(p >= 0.5).sum().over("s1", "src").cast(pl.UInt16),
    ).with_columns(p1_gap=pl.col("p1_max") - p)


TRANSITIVE = ["t_nseed", "t_name_max", "t_name_mean", "t_addr_max", "t_house_any", "t_xsrc_name_max",
              "t_name_exact_any", "t_name_vs_s1"]


def transitive_support(d: pl.DataFrame, pool: pl.DataFrame, seed_p=0.9) -> pl.DataFrame:
    """TransClean-style one-hop consistency: compare every candidate with the S1's OTHER confident
    matches (stage-1 p1 >= seed_p). A candidate whose house number / name agrees with the S1's
    other observations is supported even when it disagrees with the S1 record itself; a confident
    candidate that agrees with none of them is suspicious. d needs s1, rid, src, p1, name_tset;
    all candidates of an S1 must be in d."""
    strings = pool.select("rid", c_name="name_core", c_addr="addr_norm", c_house="house_no")
    x = d.select("s1", "rid", "src", "p1").join(strings, on="rid", how="left")
    seeds = x.filter(pl.col("p1") >= seed_p).select("s1", seed_rid="rid", seed_src="src", s_name="c_name",
                                                     s_addr="c_addr", s_house="c_house")
    x = x.drop("p1").join(seeds, on="s1").filter(pl.col("rid") != pl.col("seed_rid"))
    if len(x):
        x = x.with_columns(
            sn=pl.Series(process.cpdist(x["c_name"].to_list(), x["s_name"].to_list(), scorer=fuzz.token_set_ratio,
                                        workers=-1).astype(np.float32)),
            sa=pl.Series(process.cpdist(x["c_addr"].to_list(), x["s_addr"].to_list(), scorer=fuzz.token_set_ratio,
                                        workers=-1).astype(np.float32)))
    else:
        x = x.with_columns(sn=pl.lit(None, pl.Float32), sa=pl.lit(None, pl.Float32))
    x = x.with_columns(
        sa=pl.when((pl.col("c_addr") == "") | (pl.col("s_addr") == "")).then(None).otherwise(pl.col("sa")),
        sh=(pl.col("c_house") != "") & (pl.col("c_house") == pl.col("s_house")),
        xsrc=pl.col("src") != pl.col("seed_src"))
    sup = x.group_by("s1", "rid").agg(
        t_nseed=pl.len(), t_name_max=pl.col("sn").max(), t_name_mean=pl.col("sn").mean(),
        t_addr_max=pl.col("sa").max(), t_house_any=pl.col("sh").any().cast(pl.Float32),
        t_xsrc_name_max=pl.col("sn").filter(pl.col("xsrc")).max(),
        t_name_exact_any=(pl.col("sn") == 100).any().cast(pl.Float32))
    return d.join(sup, on=["s1", "rid"], how="left").with_columns(
        pl.col("t_nseed").fill_null(0), t_name_vs_s1=pl.col("t_name_max") - pl.col("name_tset"))


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, chunk=2_000_000) -> pl.DataFrame:
    """cand: blocking output (s1, rid, src, score_n, score_a, score, nk, ktmask, rank_*).
    s1/pool must already carry cat_df (add_name_commonness). Returns s1, rid + FEATURES."""
    from blocking import KT_BITS
    cand = cand.with_columns([pl.lit(None, dt).alias(c) for c, dt in REVERSE_COLS.items() if c not in cand.columns])
    cand = cand.with_columns([pl.lit(None, pl.Float32).alias(c) for c in ("r_best", "r_second", "r_n")
                              if c not in cand.columns])
    cand = cand.with_columns(r_gap_best=pl.col("r_best") - pl.col("rscore"),     # 0 = this S1 is the record's best
                             r_margin12=pl.col("r_best") - pl.col("r_second"),  # how clear the record's best S1 is
                             r_ratio=pl.col("rscore") / pl.col("r_best"))
    cand = cand.with_columns(pl.col("rrank_n", "rrank_a", "rrank_c").fill_null(999),
                             pl.col("rev_added").fill_null(False)).with_columns(
        rrank_best=pl.min_horizontal("rrank_n", "rrank_a", "rrank_c"))
    ctx = cand.with_columns(pl.col(REVERSE_MARGINS).cast(pl.Float32)).with_columns(
        # relative to the best candidate of the same (S1, source); null (= missing) when that best is 0
        **{f"{c}_rel": pl.when(pl.col(c).max().over("s1", "src") > 0)
           .then(pl.col(c) / pl.col(c).max().over("s1", "src")) for c in ("score", "score_n", "score_a")},
        n_cand_s1=pl.len().over("s1").cast(pl.UInt16),
        **{f"kt_{k}": (pl.col("ktmask") & v) > 0 for k, v in KT_BITS.items()},
    ).with_columns(pl.col("rank_n", "rank_a", "rank_c").fill_null(999))
    a = s1.select("s1", *[pl.col(c).alias("a_" + c) for c in REC_COLS])
    b = pool.select("rid", *[pl.col(c).alias("b_" + c) for c in REC_COLS])
    out = []
    for i in range(0, len(ctx), chunk):
        c = ctx.slice(i, chunk).join(a, on="s1").join(b, on="rid")
        c = c.with_columns(n_channels=pl.sum_horizontal(      # retrieval agreement: channels that found the pair
            [(pl.col(r) < 999).cast(pl.UInt8) for r in ("rank_n", "rank_a", "rank_c", "rrank_n", "rrank_a", "rrank_c")]))
        out.append(_block(c).select("s1", "rid", *FEATURES, *EXTRA_FEATURES))
    if not out:   # no candidates at all for this S1 chunk
        return pl.DataFrame(schema={"s1": pl.UInt32, "rid": pl.UInt32,
                                    **{f: pl.Float32 for f in FEATURES + EXTRA_FEATURES}})
    return pl.concat(out)
