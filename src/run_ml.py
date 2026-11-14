"""The ML solution on the competition match.

Everything needed is in the repository: the PathCRF code and pretrained model
(third_party/pathcrf) and our dead-ball model (trained_models/dead_ball_model.joblib, produced by
train_dead_model.py). Both were trained on external data only (DFL/IDSSE matches).

1. D: stitch the 46 ATD slots into 11 players per team (stitch.py), compute speeds, predict
   dead / alive with the dead-ball model, then drop stoppages < 4 s and restarts < 2 s.
2. H/A: PathCRF on the alive stretches, plus a ball vote: a PathCRF possession changes team
   if the ball clearly sits with the other team (nearest player <= 2 m, margin >= 0.75 m)
   on >= 60% of the frames with ball evidence (at least 1 s of evidence).
3. Minimum possession tau = 2 s. X outside the halves; D before the second-half kick-off
   (boundaries published in DATA.md / FAQ.md).

All parameters were chosen on degraded DFL matches (robust_dfl.py, ha_fusion_dfl.py).
Prints aggregates only. Output: pathcrf_out/submission_ml.csv.

Needs features_ball.csv (python src/features.py) for the ball positions.
    python src/run_ml.py            (add --cpu to force CPU)
"""
from pathlib import Path
import argparse
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATHCRF = ROOT / "third_party" / "pathcrf"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PATHCRF))

from src.check_slots import load_slots  # noqa: E402
from src.dead_dfl import frame_features, smooth  # noqa: E402
from src.eval_dfl import load_model, min_duration, pathcrf_labels  # noqa: E402
from src.ha_fusion_dfl import ball_evidence, vote  # noqa: E402
from src.paths import ATD_PATH, FEATURES, META_PATH  # noqa: E402
from src.pathcrf_adapter import OUT, PITCH_X_PATHCRF, build_tracking, describe  # noqa: E402
from src.robust_dfl import with_episodes  # noqa: E402

MODEL_PATH = ROOT / "trained_models" / "dead_ball_model.joblib"
N_FRAMES = 199_319
SECOND_HALF_EVAL_START, SECOND_HALF_KICKOFF = 110808, 113742   # DATA.md / FAQ.md
PITCH_Y = 68.0
MIN_DEAD_S, MIN_ALIVE_S = 4, 2
TAU_S = 2
VOTE_DIST_M = 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    opts = parser.parse_args()
    import joblib
    import torch
    from datatools.utils import calculate_running_features
    pd.set_option("future.no_silent_downcasting", True)
    device = "cpu" if opts.cpu or not torch.cuda.is_available() else "cuda:0"
    OUT.mkdir(exist_ok=True)

    print("1. Stitching slots, speeds, dead-ball model...")
    model_d = joblib.load(MODEL_PATH)
    slots_xy, slot_team = load_slots(ATD_PATH, META_PATH)
    table = calculate_running_features(
        build_tracking(slots_xy, slot_team, np.full(N_FRAMES, "H", dtype=object))).reset_index(drop=True)
    period = table["period_id"].to_numpy()
    frames = table["frame_id"].to_numpy()
    dead = smooth(model_d.predict_proba(frame_features(table, False))[:, 1] > 0.5, period, MIN_DEAD_S, MIN_ALIVE_S)

    print(f"2. PathCRF on {device} + ball vote...")
    warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)   # inside PathCRF
    model, args = load_model(PATHCRF, device)
    team, _ = pathcrf_labels(model, args, with_episodes(table, dead))
    team = pd.Series(team).ffill().bfill().to_numpy(dtype=object)

    feat = pd.read_csv(FEATURES)
    table["ball_x"] = feat["ball_x"].to_numpy(float)[frames] * PITCH_X_PATHCRF   # ATD coordinates are 0-1
    table["ball_y"] = feat["ball_y"].to_numpy(float)[frames] * PITCH_Y
    evidence = ball_evidence(table, VOTE_DIST_M)
    team = vote(team, evidence, dead)

    labels = np.full(N_FRAMES, "X", dtype=object)
    labels[SECOND_HALF_EVAL_START:SECOND_HALF_KICKOFF] = "D"
    labels[frames] = np.where(dead, "D", team)
    labels = min_duration(labels, TAU_S)
    out_csv = OUT / "submission_ml.csv"
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(out_csv, index=False)

    graded = labels != "X"
    print("\n3. Aggregates")
    print(f"  D on graded frames: {np.mean(labels[graded] == 'D'):.1%} (DFL reference: 40-52%)")
    print(f"  ball evidence (<= {VOTE_DIST_M:.0f} m) on {np.mean(pd.notna(evidence[~dead])):.1%} of alive frames")
    describe("ML", labels)

    print()
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(out_csv)], check=False)


if __name__ == "__main__":
    main()
