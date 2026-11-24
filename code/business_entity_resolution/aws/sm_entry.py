"""SageMaker entry point: runs the SAME src/train.py or src/inference.py inside the job.

Channels:  data  -> s3://.../student_resource/   (dataset/ + utils/validate_submission.py)
           model -> s3://.../model.tar.gz         (inference only; the bundle from a train job)
Outputs:   train -> SM_MODEL_DIR (bundle -> model.tar.gz), runs/ -> SM_OUTPUT_DATA_DIR (output.tar.gz)
           infer -> output/*.tsv + pair scores -> SM_OUTPUT_DATA_DIR
Caches go to local disk (/tmp/work) so they are not uploaded.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # project root inside the container
WORK = Path("/tmp/work")


def run(script, *args):
    cmd = [sys.executable, "-u", "-W", "ignore", str(HERE / "src" / script), *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE / "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer"], required=True)
    ap.add_argument("--config", default="configs/dev_s2.json")
    ap.add_argument("--threshold", type=float)
    a, _ = ap.parse_known_args()
    data = Path(os.environ["SM_CHANNEL_DATA"])
    # accept either <root>/dataset/{train,test} or <root>/{train,test} (shared buckets differ)
    dataset = data / "dataset" if (data / "dataset" / "train").exists() or (data / "dataset" / "test").exists() else data
    validator = next((p for p in (data / "utils" / "validate_submission.py",
                                  HERE / "aws" / "validate_submission.py") if p.exists()))
    print("dataset:", dataset, "validator:", validator, flush=True)
    out = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    if a.mode == "train":
        run("train.py", "--config", HERE / a.config, "--data-dir", dataset,
            "--work-dir", WORK, "--model-dir", os.environ["SM_MODEL_DIR"])
        for p in ("runs", "experiments.tsv"):
            src = WORK / p
            if src.is_dir():
                shutil.copytree(src, out / p, dirs_exist_ok=True)
            elif src.exists():
                shutil.copy2(src, out / p)
    else:
        model = Path("/tmp/model")
        for tgz in Path(os.environ["SM_CHANNEL_MODEL"]).glob("*.tar.gz"):
            with tarfile.open(tgz) as t:
                t.extractall(model)
        extra = ["--threshold", a.threshold] if a.threshold is not None else []
        run("inference.py", "--model-dir", model, "--data-dir", dataset, "--work-dir", WORK,
            "--out-dir", out / "output", "--validator", validator, *extra)


if __name__ == "__main__":
    main()
