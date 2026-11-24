"""Batch 4.
E-TRANS   (TransClean-style transitive consistency / one-hop S1 profile): compare each candidate with
          the S1's OTHER confident matches (stage-1 p1 >= 0.9; p1 is out-of-sample for validation).
          Stage-2-level evidence, so the control is BASE + S1_CONTEXT (stage-2 feature set).
E-XCOUNTRY diagnostic: train on one country, validate on the other (France-transfer risk).
"""
import sys, time
sys.path.insert(0, "D:/ml-challenge/lab")
sys.path.insert(0, "D:/ml-challenge/code/business_entity_resolution/src")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from harness import BASE_FEATURES, CACHE, evaluate, fit_predict, load, log
from features import S1_CONTEXT

t0 = time.time()
v, ids, truth, _ = load()
cols = ["entity_id", "name_core", "addr_norm", "house_no", "country"]
a = (pl.scan_parquet(CACHE + "norm_n4_train_s1.parquet").select(cols)
     .filter(pl.col("entity_id").is_in(v["s1_id"].unique().implode())).collect())
b = pl.concat([pl.scan_parquet(CACHE + f"norm_n4_train_s{i}.parquet").select(cols[:4])
               .filter(pl.col("entity_id").is_in(v["cand_id"].unique().implode())).collect() for i in (2, 3)])
v = v.join(a.select(s1_id="entity_id", country="country"), on="s1_id").join(
    b.rename({"entity_id": "cand_id", "name_core": "c_name", "addr_norm": "c_addr", "house_no": "c_house"}), on="cand_id")

# ---- one-hop support: every candidate x the S1's confident matches (seeds), excluding itself
SEED = 0.9
seeds = v.filter(pl.col("p1") >= SEED).select("s1", seed_rid="rid", seed_src="src", s_name="c_name",
                                               s_addr="c_addr", s_house="c_house")
x = (v.select("s1", "rid", "src", "c_name", "c_addr", "c_house").join(seeds, on="s1")
     .filter(pl.col("rid") != pl.col("seed_rid")))
print(f"pairs {len(v):,}  seed links {len(x):,}  ({time.time()-t0:.0f}s)", flush=True)
x = x.with_columns(
    sn=pl.Series(process.cpdist(x["c_name"].to_list(), x["s_name"].to_list(), scorer=fuzz.token_set_ratio, workers=8)),
    sa=pl.Series(process.cpdist(x["c_addr"].to_list(), x["s_addr"].to_list(), scorer=fuzz.token_set_ratio, workers=8)),
).with_columns(
    sa=pl.when((pl.col("c_addr") == "") | (pl.col("s_addr") == "")).then(None).otherwise(pl.col("sa")),
    sh=(pl.col("c_house") != "") & (pl.col("c_house") == pl.col("s_house")),
    xsrc=pl.col("src") != pl.col("seed_src"))
sup = x.group_by("s1", "rid").agg(
    t_nseed=pl.len(), t_name_max=pl.col("sn").max(), t_name_mean=pl.col("sn").mean(),
    t_addr_max=pl.col("sa").max(), t_house_any=pl.col("sh").any().cast(pl.Float32),
    t_xsrc_name_max=pl.col("sn").filter(pl.col("xsrc")).max(),
    t_name_exact_any=(pl.col("sn") == 100).any().cast(pl.Float32))
v = v.join(sup, on=["s1", "rid"], how="left").with_columns(
    pl.col("t_nseed").fill_null(0),
    t_name_vs_s1=pl.col("t_name_max") - pl.col("name_tset"))        # >0: closer to the S1's observed variants than to S1
TRANS = ["t_nseed", "t_name_max", "t_name_mean", "t_addr_max", "t_house_any", "t_xsrc_name_max",
         "t_name_exact_any", "t_name_vs_s1"]
print(f"prepared in {time.time()-t0:.0f}s", flush=True)


from features import house_change_features, HOUSE_CHANGE, REVERSE_MARGINS
import glob
a2 = (pl.scan_parquet(CACHE + "norm_n4_train_s1.parquet").select(s1_id="entity_id", ha="house_no")
      .filter(pl.col("s1_id").is_in(v["s1_id"].unique().implode())).collect())
v = v.join(a2, on="s1_id", how="left").with_columns(pl.col("ha").fill_null(""), hb=pl.col("c_house").fill_null(""))
v = v.with_columns(**house_change_features(pl.col("ha"), pl.col("hb")))
rev = (pl.scan_parquet(sorted(glob.glob(CACHE + "reverse_*_train.parquet"))[-1]).select("rid", "rscore")
       .filter(pl.col("rid").is_in(v["rid"].unique().implode())).collect())
top = rev.group_by("rid").agg(r_best=pl.col("rscore").max(), r_second=pl.col("rscore").top_k(2).min(), r_n=pl.len())
v = v.join(top, on="rid", how="left").with_columns(r_gap_best=pl.col("r_best") - pl.col("rscore"),
    r_margin12=pl.col("r_best") - pl.col("r_second"), r_ratio=pl.col("rscore") / pl.col("r_best"))

import numpy as np
from rapidfuzz.distance import Levenshtein
BASEHR = BASE_FEATURES + S1_CONTEXT + HOUSE_CHANGE + REVERSE_MARGINS

# ---------- X1: cross-source corroboration (opposite-source, plausibly-same-S1 candidates only)
v = v.with_columns(p1_rank_in_src=pl.col("p1").rank("ordinal", descending=True).over("s1", "src"))
opp = v.filter((pl.col("p1_rank_in_src") <= 3) & (pl.col("p1") >= 0.5)).select(
    "s1", o_rid="rid", o_src="src", o_p1="p1", o_name="c_name", o_addr="c_addr", o_house="c_house")
xx = v.select("s1", "rid", "src", "c_name", "c_addr", "c_house").join(opp, on="s1").filter(pl.col("src") != pl.col("o_src"))
print(f"cross-source links {len(xx):,}", flush=True)
xx = xx.with_columns(
    xn=pl.Series(process.cpdist(xx["c_name"].to_list(), xx["o_name"].to_list(), scorer=fuzz.token_set_ratio, workers=8)),
    xa=pl.Series(process.cpdist(xx["c_addr"].to_list(), xx["o_addr"].to_list(), scorer=fuzz.token_set_ratio, workers=8)),
    xh=pl.Series(process.cpdist(xx["c_house"].to_list(), xx["o_house"].to_list(), scorer=Levenshtein.distance, workers=8)))
xx = xx.with_columns(
    xa=pl.when((pl.col("c_addr") == "") | (pl.col("o_addr") == "")).then(None).otherwise(pl.col("xa")),
    xh=pl.when((pl.col("c_house") == "") | (pl.col("o_house") == "")).then(None).otherwise(pl.col("xh")))
top_o = pl.col("o_p1") == pl.col("o_p1").max()
agg = xx.group_by("s1", "rid").agg(
    x_n=pl.len(), x_name_max=pl.col("xn").max(), x_addr_max=pl.col("xa").max(),
    x_house_eq=(pl.col("xh") == 0).any().cast(pl.Float32), x_house_ed_min=pl.col("xh").min(),
    x_best_p1=pl.col("o_p1").max(), x_support=(pl.col("o_p1") * pl.col("xn") / 100).max(),
    x_contra=((pl.col("o_p1") >= 0.9) & (pl.col("xn") < 60) & (pl.col("xa").fill_null(0) < 60)).any().cast(pl.Float32))
v = v.join(agg, on=["s1", "rid"], how="left").with_columns(pl.col("x_n").fill_null(0))
X1 = ["x_n", "x_name_max", "x_addr_max", "x_house_eq", "x_house_ed_min", "x_best_p1", "x_support", "x_contra"]

# ---------- X2: profile-only with STRICT seeds (p1 >= 0.95), same + cross source
s95 = v.filter(pl.col("p1") >= 0.95).select("s1", seed_rid="rid", s_name="c_name", s_house="c_house")
y = v.select("s1", "rid", "c_name", "c_house").join(s95, on="s1").filter(pl.col("rid") != pl.col("seed_rid"))
y = y.with_columns(sn=pl.Series(process.cpdist(y["c_name"].to_list(), y["s_name"].to_list(), scorer=fuzz.token_set_ratio, workers=8)))
p95 = y.group_by("s1", "rid").agg(p95_n=pl.len(), p95_name_max=pl.col("sn").max(),
      p95_house_any=((pl.col("c_house") != "") & (pl.col("c_house") == pl.col("s_house"))).any().cast(pl.Float32))
v = v.join(p95, on=["s1", "rid"], how="left").with_columns(pl.col("p95_n").fill_null(0))
X2 = ["p95_n", "p95_name_max", "p95_house_any"]
print(f"prepared {time.time()-t0:.0f}s", flush=True)

S2F = BASE_FEATURES + S1_CONTEXT
oofs = {}
for seed in (0, 1):
    oofs[f"x1_{seed}"] = fit_predict(v, S2F + X1, seed=seed)
    log({**evaluate(v, oofs[f"x1_{seed}"], ids, truth, f"stage2ctl+X1_xsrc_s{seed}"), "exp": "xsrc"})
for seed in (0, 1):
    log({**evaluate(v, fit_predict(v, S2F + X2, seed=seed), ids, truth, f"stage2ctl+X2_profile95_s{seed}"), "exp": "xsrc"})
for seed in (0, 1):
    log({**evaluate(v, fit_predict(v, BASEHR + TRANS + X1, seed=seed), ids, truth, f"fullstack+X1_s{seed}"), "exp": "xsrc"})

# ---------- flip analysis: which predictions does X1 change, and did cross-source evidence supply them?
ctl = fit_predict(v, S2F, seed=0)
def decide(oof):
    r = evaluate(v, oof, ids, truth, "tmp", slices=False)
    return oof >= r["t"]
a, b = decide(ctl), decide(oofs["x1_0"])
lab = v["label"].to_numpy() == 1
fx = v.with_columns(ctl_ok=pl.Series(a == lab), x1_ok=pl.Series(b == lab))
fixed = fx.filter(~pl.col("ctl_ok") & pl.col("x1_ok")); broke = fx.filter(pl.col("ctl_ok") & ~pl.col("x1_ok"))
def prof(d):
    return {"n": len(d), "pos": int(d["label"].sum()), "addr_empty": round(d["b_addr_empty"].mean(), 3) if len(d) else None,
            "has_xsrc_support(x_name>=90 & x_best_p1>=0.9)": round(((d["x_name_max"].fill_null(0) >= 90) & (d["x_best_p1"].fill_null(0) >= 0.9)).mean(), 3) if len(d) else None,
            "x_contra": round(d["x_contra"].fill_null(0).mean(), 3) if len(d) else None}
log({"name": "X1 flip analysis (seed 0)", "fixed": prof(fixed), "broke": prof(broke), "exp": "xsrc"})
print(f"done {time.time()-t0:.0f}s")
