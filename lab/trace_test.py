"""Controlled end-to-end trace through the REAL pipeline functions + the frozen baseline models.

A small hand-built universe with known answers exercises: exact match, name/address/house corruption,
nearby house numbers, native script, domain-as-name, legal-form decoy, singleton with look-alike,
two S1 competing for one empty-address record, a French record, S2/S3 disagreement.
Checks: recall of true pairs into candidates, feature schema, NaN/inf, determinism (two runs),
exclusivity (each S2/S3 at most once), official validator on the written TSVs.
Uses a temporary work dir (no real cache is touched).
"""
import json, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, "D:/ml-challenge/code/business_entity_resolution/src")
sys.stdout.reconfigure(encoding="utf-8")
import lightgbm as lgb
import numpy as np
import polars as pl
import config, decision, submission
from blocking import build_index, build_keys, candidates, reverse_ranks, reverse_stats
from features import (EXTRA_FEATURES, FEATURES, S1_CONTEXT, TRANSITIVE, add_name_commonness,
                      pair_features, s1_context, transitive_support)
from normalize import normalize_records

BASE = Path("D:/ml-challenge/artifacts/baseline_best")
bundle = json.load(open(BASE / "bundle.json"))
b1, b2 = lgb.Booster(model_file=str(BASE / "model.txt")), lgb.Booster(model_file=str(BASE / "model2.txt"))

S1 = [  # entity_id, name, address, country, case
    ("S1-1", "Grand Classic Transition", "12417 143rd Avenue, Puyallup, WA", "US", "exact + typo + legal decoy"),
    ("S1-2", "Tubbs Classic Supply", "2405 Lambert Drive, Toledo, OH", "US", "house corruption / nearby number"),
    ("S1-3", "Raj Investments LLP", "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu", "India", "native script"),
    ("S1-4", "Twyla's Liquor Corp", "320 Flannery Lane, Silver Spring, MD", "US", "singleton with look-alike"),
    ("S1-5", "Sheetal Holdings", "Flat 201, Wing B, Andheri, Mumbai, Maharashtra", "India", "competition A"),
    ("S1-6", "Sheetal Holdings", "12 MG Road, Pune, Maharashtra", "India", "competition B"),
    ("S1-7", "ZNB Club SARL", "5 bis Rue Pierre Dignac, La Teste-de-Buch, Nouvelle-Aquitaine", "France", "French record"),
    ("S1-8", "Maure Williams Colombier Inc", "85 Wayne Avenue, Ticonderoga, NY", "US", "domain name"),
]
POOL = [  # entity_id, name, address, country, true S1 (or None)
    ("S2-11", "GRAND CLASSIC TRANSITION", "12417 143RD AVE, PUYALLUP, WA", "US", "S1-1"),
    ("S3-12", "Grand Classic Transighdn", "12417 143rd Ave, Puyallup, Washington", "US", "S1-1"),
    ("S3-13", "Grand Classic Transition LLC", "12417 143rd Avenue, Puyallup, Washington", "US", None),
    ("S2-21", "TUBBS CLASSIC SUPPLY", "2405 LAMBERT DR, TOLEDO, OH", "US", "S1-2"),
    ("S3-22", "Tubbs Classic Supply LLC", "3405 Lambert Dr, Toledo, Ohio", "US", "S1-2"),
    ("S2-31", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu", "India", "S1-3"),
    ("S3-32", "Raj Investments எல்எல்பி", "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN", "India", "S1-3"),
    ("S2-41", "Twyla's Center", "16106 Necoridge Drive, Houston, TX", "US", None),
    ("S3-51", "Sheetal Holdings", "", "India", "S1-5"),
    ("S2-52", "SHEETAL HOLDINGS", "FLAT 201, WING B, ANDHERI, MUMBAI, Maharashtra", "India", "S1-5"),
    ("S2-61", "Sheetal Holdings Care", "12 M.G. ROAD, PUNE, MH", "India", "S1-6"),
    ("S2-71", "ZNB CLUB SARL", "5 R PIERRE DIGNAC, LA TESTE-DE-BUCH, Gironde", "France", "S1-7"),
    ("S3-72", "Znb Club", "5 bis Rue Pierre Dignac, La Teste-de-buch", "France", "S1-7"),
    ("S3-81", "maurewilliamscolombier.com", "Wayne Ave, Ticonderoga Townshiip, New York", "US", "S1-8"),
    ("S2-82", "Maure Wilblims Colombier Inc", "", "US", "S1-8"),
]
rng = np.random.default_rng(0)                  # filler records so IDF is meaningful
WORDS = ["alpha", "beta", "delta", "omega", "prime", "summit", "vista", "nova", "zen", "apex", "orbit", "lumen"]
STREETS = ["Oak", "Pine", "Maple", "Cedar", "Elm", "Lake", "Hill", "Park"]
for i in range(240):
    c = ["US", "India", "France"][i % 3]
    nm = f"{rng.choice(WORDS).title()} {rng.choice(WORDS).title()} {rng.choice(['Inc', 'LLC', 'Pvt Ltd', 'SARL'])}"
    ad = f"{rng.integers(1, 9999)} {rng.choice(STREETS)} Street, Town{rng.integers(1, 30)}"
    POOL.append((f"S{2 + i % 2}-9{i:03d}", nm, ad, c, None))


def run_once(tmp):
    cfg = config.load("D:/ml-challenge/code/business_entity_resolution/configs/rev63_hrt.json")
    cfg.work_dir = str(tmp)
    df1 = pl.DataFrame([r[:4] for r in S1], schema=["entity_id", "business_name", "business_address", "country"], orient="row")
    s1 = normalize_records(df1, 1)
    pool = pl.concat([normalize_records(pl.DataFrame([r[:4] for r in POOL if r[0].startswith(p)],
                                                     schema=["entity_id", "business_name", "business_address", "country"],
                                                     orient="row"), int(p[1])) for p in ("S2", "S3")])
    countries = pl.Enum(sorted(set(s1["country"]) | set(pool["country"])))
    cc = pl.col("country").cast(countries).to_physical().cast(pl.UInt8)
    s1 = s1.with_row_index("s1").with_columns(cc=cc)
    pool = pool.with_row_index("rid").with_columns(cc=cc)
    s1, pool = add_name_commonness(s1, pool)
    k = build_keys(pool, cfg.blocking.key_types).join(pool.select("rid", "src", "cc"), on="rid")
    k = k.join(k.group_by("key").agg(df=pl.len().cast(pl.UInt32)), on="key")
    idx = build_index(k, pool, cfg)
    rev = reverse_ranks(cfg, "trace", s1, pool)
    cand = candidates(cfg, s1, idx, (rev, reverse_stats(rev)))
    f = pair_features(cand, s1, pool)
    f = f.join(s1.select("s1", s1_id="entity_id"), on="s1").join(pool.select("rid", cand_id="entity_id"), on="rid")
    f = f.with_columns(p1=pl.Series(b1.predict(f.select(bundle["features"]).cast(pl.Float32).to_numpy()), dtype=pl.Float32))
    f = transitive_support(s1_context(f), pool)
    f = f.with_columns(p=pl.Series(b2.predict(f.select(bundle["stage2"]["features"]).cast(pl.Float32).to_numpy()),
                                   dtype=pl.Float32))
    return f.sort("s1_id", "cand_id"), s1


tmp1, tmp2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
f, s1 = run_once(tmp1)
f_again, _ = run_once(tmp2)
truth = pl.DataFrame([(r[4], r[0]) for r in POOL if r[4]], schema=["s1_id", "cand_id"], orient="row")
checks = {}
# 1 candidate recall on the toy universe
hit = truth.join(f.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="semi")
checks["all true pairs in candidates"] = (len(hit) == len(truth), f"{len(hit)}/{len(truth)}")
# 2 schema
need = set(FEATURES + EXTRA_FEATURES + S1_CONTEXT + TRANSITIVE + bundle["features"] + bundle["stage2"]["features"])
checks["feature schema complete"] = (need <= set(f.columns), sorted(need - set(f.columns)))
# 3 NaN / inf in model inputs
num = f.select([c for c in bundle["stage2"]["features"]]).cast(pl.Float64)
inf = sum(int(num[c].is_infinite().sum()) for c in num.columns)
nan = sum(int(num[c].is_nan().sum()) for c in num.columns)
checks["no inf/NaN in stage-2 inputs (nulls allowed)"] = (inf == 0 and nan == 0, f"inf={inf} nan={nan}")
# 4 determinism
checks["deterministic (two independent runs identical)"] = (f.select("s1_id", "cand_id", "p1", "p").equals(
    f_again.select("s1_id", "cand_id", "p1", "p")), "")
# 5 no duplicate pairs
checks["no duplicate candidate pairs"] = (f.select("s1_id", "cand_id").is_duplicated().sum() == 0, "")
# decision with the baseline threshold + exclusivity
t = bundle["threshold"]
m = decision.apply_threshold(decision.exclusive(f.select("s1_id", "cand_id", "p")), t)
checks["exclusivity: each S2/S3 at most one S1"] = (m["cand_id"].is_duplicated().sum() == 0, "")
checks["matches subset of candidates"] = (m.join(f, on=["s1_id", "cand_id"], how="anti").is_empty(), "")
# 6 official validator on the toy output
tdir = tmp1 / "test"; tdir.mkdir()
for src, rows in (("1", S1), ("2", [r for r in POOL if r[0].startswith("S2")]), ("3", [r for r in POOL if r[0].startswith("S3")])):
    pl.DataFrame([r[:4] for r in rows], schema=["entity_id", "business_name", "business_address", "country"],
                 orient="row").write_csv(tdir / f"test_source{src}.tsv", separator="\t")
submission.write_outputs(tmp1 / "out", s1["entity_id"], f.select("s1_id", "cand_id", score="p"), m)
rc = subprocess.run([sys.executable, "D:/ml-challenge/student_resource/utils/validate_submission.py", "--matching",
                     str(tmp1 / "out/matching_results.tsv"), "--candidate", str(tmp1 / "out/candidate_pairs.tsv"),
                     "--test-dir", str(tdir), "--check-ids"], capture_output=True, text=True, encoding="utf-8")
checks["official validator PASS (with --check-ids)"] = (rc.returncode == 0, rc.stdout.strip().splitlines()[-1])

print("=== CHECKS ===")
for k_, (ok, info) in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {k_}  {info}")
case = {r[0]: r[4] for r in S1}
lab = truth.with_columns(label=pl.lit(1))
tr = (f.join(lab, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
      .join(m.select("s1_id", "cand_id", matched=pl.lit(True)), on=["s1_id", "cand_id"], how="left")
      .filter((pl.col("label") == 1) | (pl.col("p") > 0.05) | pl.col("matched").is_not_null())
      .with_columns(case=pl.col("s1_id").replace_strict(case), matched=pl.col("matched").fill_null(False))
      .select("case", "s1_id", "cand_id", "label", "rank_c", "rrank_best", "rev_added", pl.col("name_tset").round(0),
              pl.col("addr_tset").round(0), "house_eq", pl.col("house_absdiff"), pl.col("r_gap_best").round(1),
              pl.col("p1").round(3), "p1_rank", "t_nseed", "t_house_any", pl.col("p").round(3), "matched")
      .sort("s1_id", "p", descending=[False, True]))
with pl.Config(tbl_rows=60, tbl_width_chars=260, fmt_str_lengths=30, tbl_hide_dataframe_shape=True):
    print(tr)
wrong = tr.filter(pl.col("label").cast(pl.Boolean) != pl.col("matched"))
print(f"decision errors on toy universe: {len(wrong)} (FN={int(((wrong['label'] == 1)).sum())}, FP={int(((wrong['label'] == 0)).sum())})")
