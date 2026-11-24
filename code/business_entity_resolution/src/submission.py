"""Writers for matching_results.tsv / candidate_pairs.tsv and the official validator call.

Format (problem statement + utils/validate_submission.py):
  * UTF-8, TAB-separated, LF line endings (the validator strips only "\n"; a CRLF
    file would glue "\r" onto the last ID of every row)
  * exact headers, one row per test S1 in file order, empty list for no match
  * comma-joined S2-/S3- IDs, no duplicates, matches must be a subset of candidates
"""
import subprocess
import sys
from pathlib import Path

import polars as pl

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def write_id_lists(path, header, s1_ids: pl.Series, pairs: pl.DataFrame):
    """pairs: long (s1_id, cand_id[, score]); IDs within a list ordered by score desc when given."""
    assert pairs.filter(~pl.col("cand_id").str.contains(r"^S[23]-")).is_empty(), "only S2-/S3- IDs allowed"
    if "score" in pairs.columns:
        pairs = pairs.sort(["s1_id", "score", "cand_id"], descending=[False, True, False])
    lists = pairs.unique(["s1_id", "cand_id"], maintain_order=True).group_by("s1_id", maintain_order=True).agg(
        ids=pl.col("cand_id").str.join(","))
    rows = pl.DataFrame({"s1_id": s1_ids}).join(lists, on="s1_id", how="left").fill_null("")
    assert rows["s1_id"].n_unique() == len(rows) == len(s1_ids), "duplicate or missing S1 rows"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1, ids in rows.iter_rows():
            f.write(f"{s1}\t{ids}\n")


def write_outputs(out_dir, s1_ids, cand_pairs, match_pairs):
    out = Path(out_dir)
    assert match_pairs.join(cand_pairs, on=["s1_id", "cand_id"], how="anti").is_empty(), \
        "every match must be a candidate"
    write_id_lists(out / "candidate_pairs.tsv", CAND_HEADER, s1_ids, cand_pairs)
    write_id_lists(out / "matching_results.tsv", MATCH_HEADER, s1_ids, match_pairs)


def validate(out_dir, test_dir, validator):
    """Run the official utils/validate_submission.py; returns its exit code (0 = PASS)."""
    out = Path(out_dir)
    cmd = [sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
           "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(test_dir)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    import tempfile
    s1 = pl.Series(["S1-1", "S1-2", "S1-3"])
    cand = pl.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-1", "S1-2"], "cand_id": ["S2-5", "S3-7", "S2-5", "S3-4"],
                         "score": [0.9, 0.8, 0.9, 0.1]})
    match = cand.filter(pl.col("score") > 0.5)
    with tempfile.TemporaryDirectory() as d:
        write_outputs(d, s1, cand, match)
        raw = (Path(d) / "matching_results.tsv").read_bytes()
        assert b"\r" not in raw
        assert raw.decode().splitlines() == ["source1_entity_id\tmatched_entity_ids",
                                             "S1-1\tS2-5,S3-7", "S1-2\t", "S1-3\t"], raw
    print("submission self-check OK")
