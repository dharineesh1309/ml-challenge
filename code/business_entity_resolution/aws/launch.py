"""Launch the pipeline on SageMaker (run on your laptop; needs `aws configure` done).

    python aws/launch.py upload --bucket B                     # dataset + validator -> s3://B/ber/student_resource/
    python aws/launch.py train  --bucket B --role ROLE_ARN [--config configs/dev_s2.json] [--instance ml.m5.24xlarge]
    python aws/launch.py infer  --bucket B --role ROLE_ARN --model-s3 s3://B/ber/jobs/<job>/output/model.tar.gz
    # shared bucket: read an existing dataset in place, write only under our own prefix
    python aws/launch.py train  --bucket B --prefix ber-me --data-s3 s3://B/path/to/dataset --role ROLE_ARN
    python aws/launch.py fetch  --s3 s3://B/ber/jobs/<job>/output/output.tar.gz --dest ../../output_sm

Jobs run aws/sm_entry.py (same src/train.py / src/inference.py as local) in the PyTorch CPU
container; requirements.txt is pip-installed by the container. Each job is billed per second.
"""
import argparse
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_LOCAL = ROOT.parent.parent / "student_resource"


def estimator(a, mode, extra_hp=None):
    from sagemaker.pytorch import PyTorch
    return PyTorch(
        entry_point="aws/sm_entry.py", source_dir=str(ROOT), role=a.role,
        framework_version="2.5", py_version="py311", instance_type=a.instance, instance_count=1,
        volume_size=200, max_run=6 * 3600, output_path=f"s3://{a.bucket}/{a.prefix}/jobs",
        code_location=f"s3://{a.bucket}/{a.prefix}/code",
        base_job_name=f"ber-{mode}", hyperparameters={"mode": mode, "config": a.config, **(extra_hp or {})},
        disable_profiler=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "train", "infer", "fetch"])
    ap.add_argument("--bucket")
    ap.add_argument("--role")
    ap.add_argument("--config", default="configs/dev_s2.json")
    ap.add_argument("--instance", default="ml.m5.24xlarge")
    ap.add_argument("--model-s3")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--s3")
    ap.add_argument("--dest", default=str(ROOT.parent.parent / "output_sm"))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--prefix", default="ber", help="our own folder in the (shared) bucket; jobs write only here")
    ap.add_argument("--data-s3", help="existing dataset location (read-only), holding train/ + test/ "
                                      "or dataset/{train,test}; default s3://BUCKET/PREFIX/student_resource")
    a = ap.parse_args()
    data_s3 = a.data_s3 or f"s3://{a.bucket}/{a.prefix}/student_resource"

    if a.cmd == "upload":
        for sub in ("dataset", "utils"):
            subprocess.run(["aws", "s3", "sync", str(DATA_LOCAL / sub), f"{data_s3}/{sub}",
                            "--exclude", "*.DS_Store"], check=True)
    elif a.cmd == "train":
        est = estimator(a, "train")
        est.fit({"data": data_s3}, wait=a.wait, job_name=f"ber-train-{datetime.now():%m%d-%H%M%S}")
        print("started:", est.latest_training_job.name)
    elif a.cmd == "infer":
        hp = {"threshold": a.threshold} if a.threshold is not None else {}
        est = estimator(a, "infer", hp)
        est.fit({"data": data_s3, "model": a.model_s3}, wait=a.wait,
                job_name=f"ber-infer-{datetime.now():%m%d-%H%M%S}")
        print("started:", est.latest_training_job.name)
    elif a.cmd == "fetch":
        dest = Path(a.dest)
        dest.mkdir(parents=True, exist_ok=True)
        local = dest / Path(a.s3).name
        subprocess.run(["aws", "s3", "cp", a.s3, str(local)], check=True)
        with tarfile.open(local) as t:
            t.extractall(dest)
        print("extracted to", dest)


if __name__ == "__main__":
    main()
