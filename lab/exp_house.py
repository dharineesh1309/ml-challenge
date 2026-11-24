"""E-HOUSE: house-number change-type features vs control (seed 0 / seed 1 for noise)."""
import sys, time
sys.path.insert(0, "D:/ml-challenge/lab")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import polars as pl
from harness import BASE_FEATURES, CACHE, evaluate, fit_predict, load, log

t0 = time.time()
v, ids, truth, _ = load()
a = (pl.scan_parquet(CACHE + "norm_n4_train_s1.parquet").select(s1_id="entity_id", ha="house_no")
     .filter(pl.col("s1_id").is_in(v["s1_id"].unique().implode())).collect())
b = pl.concat([pl.scan_parquet(CACHE + f"norm_n4_train_s{i}.parquet").select(cand_id="entity_id", hb="house_no")
               .filter(pl.col("cand_id").is_in(v["cand_id"].unique().implode())).collect() for i in (2, 3)])
v = v.join(a, on="s1_id", how="left").join(b, on="cand_id", how="left")

# house-number change type. Noise in true matches = a random digit changed / a digit dropped;
# look-alike negatives = NEARBY house numbers. Edit distance cannot tell these apart.
ha, hb = v["ha"].fill_null("").to_list(), v["hb"].fill_null("").to_list()
n = len(ha)
diff = np.full(n, np.nan, np.float32); rel = np.full(n, np.nan, np.float32)
ham = np.full(n, np.nan, np.float32); first = np.full(n, np.nan, np.float32); transp = np.zeros(n, np.float32)
for i, (x, y) in enumerate(zip(ha, hb)):
    if not x or not y or x == y or len(x) > 9 or len(y) > 9:
        continue
    xi, yi = int(x), int(y)
    diff[i] = abs(xi - yi)
    rel[i] = abs(xi - yi) / max(xi, yi)
    if len(x) == len(y):
        d = [k for k in range(len(x)) if x[k] != y[k]]
        ham[i] = len(d)
        first[i] = d[0] / len(x)                       # 0 = leading digit changed, ~1 = last digit
        transp[i] = float(sorted(x) == sorted(y))
v = v.with_columns(house_absdiff=pl.Series(diff), house_reldiff=pl.Series(rel), house_hamming=pl.Series(ham),
                   house_first_diff_pos=pl.Series(first), house_transposed=pl.Series(transp))
HOUSE = ["house_absdiff", "house_reldiff", "house_hamming", "house_first_diff_pos", "house_transposed"]
print(f"prepared {len(v):,} pairs in {time.time()-t0:.0f}s", flush=True)

for seed in (0, 1):
    log({**evaluate(v, fit_predict(v, BASE_FEATURES, seed=seed), ids, truth, f"control_s{seed}"), "exp": "house"})
for seed in (0, 1):
    log({**evaluate(v, fit_predict(v, BASE_FEATURES + HOUSE, seed=seed), ids, truth, f"+house_s{seed}"), "exp": "house"})
print(f"done {time.time()-t0:.0f}s")
