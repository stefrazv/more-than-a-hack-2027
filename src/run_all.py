"""Reproduce the Raumdeuters submission with one command.

    python src/run_all.py            (add --cpu to force PathCRF onto the CPU)

Steps:
  1. src/features.py  -- ball position per frame from the raw ATD (features_ball.csv)
  2. src/run_ml.py    -- dead-ball model + PathCRF + ball vote (pathcrf_out/submission_ml.csv)
  3. copy it to Raumdeuters.csv, validate it with the official validator, print its SHA256

The result is identical on CPU and GPU (same SHA256). Needs the environment from
requirements-solution.txt and the competition data unpacked in data/ (see data/README.md).
"""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import OUT_DIR  # noqa: E402

TEAM_NAME = "Raumdeuters"
ML_OUTPUT = OUT_DIR / "pathcrf_out" / "submission_ml.csv"
FINAL = OUT_DIR / f"{TEAM_NAME}.csv"


def run(command):
    print("\n>>", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="run PathCRF on the CPU even if a GPU is available")
    opts = parser.parse_args()

    run([sys.executable, "src/features.py"])
    run([sys.executable, "src/run_ml.py"] + (["--cpu"] if opts.cpu else []))

    shutil.copyfile(ML_OUTPUT, FINAL)
    run([sys.executable, "evaluation/validate_submission.py", str(FINAL)])

    digest = hashlib.sha256(FINAL.read_bytes()).hexdigest().upper()
    print(f"\nSubmission: {FINAL}")
    print(f"SHA256:     {digest}")


if __name__ == "__main__":
    main()