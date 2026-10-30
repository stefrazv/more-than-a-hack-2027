from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "src/features.py",
    "src/dead_ball.py",
    "src/possession.py",
    "src/predict.py",
]

SUBMISSION = "submission_v5.csv"


def run(command):
    print("\n>>", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    for step in STEPS:
        run([sys.executable, step])

    run([sys.executable, "evaluation/validate_submission.py", SUBMISSION])


if __name__ == "__main__":
    main()