"""Soluția ML pe meciul nostru (versiunea aleasă pe DFL stricat).

1. modelul de D se antrenează pe cele 7 meciuri DFL STRICATE și lipite la loc
   (degrade.py + stitch.py), ca să semene cu ce primește pe meciul nostru;
2. pe meciul nostru: sloturile se lipesc în 11 jucători pe echipă (stitch.py),
   modelul prezice "oprit / în joc", apoi netezire (opriri >= 4 s, reluări >= 2 s);
3. H/A = PathCRF pe porțiunile în joc + vot cu mingea: o posesie PathCRF își schimbă
   echipa dacă mingea o vede clar (<= 2 m, avantaj >= 0,75 m) la cealaltă echipă
   în >= 60% din cadrele cu dovadă (minimum 1 s);
4. posesie minimă tau = 2 s; în afara reprizelor X; înainte de lovitura de început
   a reprizei a doua D (limită publicată în DATA.md/FAQ.md).

Toate alegerile au fost făcute pe DFL stricat (ha_fusion_dfl.py, robust_dfl.py).
Afișează doar agregate. Fișiere: pathcrf_out/submission_ml.csv, pathcrf_out/dead_model.joblib.

Înainte: python src/run_all.py (are nevoie de submission_v5.csv și features_ball.csv).
    E:\\mtah\\pathcrf\\.venv\\Scripts\\python.exe src\\run_ml.py --pathcrf E:\\mtah\\pathcrf
"""
from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.check_slots import load_slots  # noqa: E402
from src.dead_dfl import frame_features, smooth  # noqa: E402
from src.eval_dfl import ground_truth, load_model, min_duration, pathcrf_labels  # noqa: E402
from src.ha_fusion_dfl import ball_evidence, vote  # noqa: E402
from src.paths import ATD_PATH, FEATURES, META_PATH, SUBMISSION  # noqa: E402
from src.pathcrf_adapter import OUT, PITCH_X_PATHCRF, build_tracking, describe  # noqa: E402
from src.robust_dfl import degraded_table, load, train, with_episodes  # noqa: E402

N_FRAMES = 199_319
SECOND_HALF_EVAL_START, SECOND_HALF_KICKOFF = 110808, 113742   # DATA.md / FAQ.md
PITCH_Y = 68.0
MIN_DEAD_S, MIN_ALIVE_S = 4, 2
TAU_S = 2
VOTE_DIST_M = 2.0
DEGRADE_SEED = 47


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathcrf", required=True)
    parser.add_argument("--cpu", action="store_true")
    opts = parser.parse_args()
    sys.path.insert(0, str(opts.pathcrf))
    import joblib
    import torch
    from datatools.utils import calculate_running_features
    pd.set_option("future.no_silent_downcasting", True)
    device = "cpu" if opts.cpu or not torch.cuda.is_available() else "cuda:0"
    OUT.mkdir(exist_ok=True)

    print("1. Modelul de D pe cele 7 meciuri DFL stricate și lipite...")
    folder = Path(opts.pathcrf) / "data" / "sportec" / "tracking_processed"
    feats, labels = [], []
    for path in sorted(folder.glob("*.parquet")):
        df = load(folder, path.stem)
        feats.append(frame_features(degraded_table(df, DEGRADE_SEED, calculate_running_features), False))
        labels.append(ground_truth(df)[0] == "D")
    model_d = train(feats, labels)
    joblib.dump(model_d, OUT / "dead_model.joblib")

    print("2. Meciul nostru: lipirea sloturilor, vitezele, D...")
    slots_xy, slot_team = load_slots(ATD_PATH, META_PATH)
    table = calculate_running_features(
        build_tracking(slots_xy, slot_team, np.full(N_FRAMES, "H", dtype=object))).reset_index(drop=True)
    period = table["period_id"].to_numpy()
    frames = table["frame_id"].to_numpy()
    dead = smooth(model_d.predict_proba(frame_features(table, False))[:, 1] > 0.5, period, MIN_DEAD_S, MIN_ALIVE_S)

    print("3. PathCRF + vot cu mingea...")
    model, args = load_model(opts.pathcrf, device)
    team, _ = pathcrf_labels(model, args, with_episodes(table, dead))
    team = pd.Series(team).ffill().bfill().to_numpy(dtype=object)

    feat = pd.read_csv(FEATURES)
    table["ball_x"] = feat["ball_x"].to_numpy(float)[frames] * PITCH_X_PATHCRF   # ATD: coordonate 0-1
    table["ball_y"] = feat["ball_y"].to_numpy(float)[frames] * PITCH_Y
    evidence = ball_evidence(table, VOTE_DIST_M)
    team = vote(team, evidence, dead)

    # 4. fișierul complet
    v5 = pd.read_csv(SUBMISSION)["label"].to_numpy(dtype=object)
    labels_out = np.full(N_FRAMES, "X", dtype=object)
    labels_out[SECOND_HALF_EVAL_START:SECOND_HALF_KICKOFF] = "D"
    labels_out[frames] = np.where(dead, "D", team)
    labels_out = min_duration(labels_out, TAU_S)
    out_csv = OUT / "submission_ml.csv"
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels_out}).to_csv(out_csv, index=False)

    graded = labels_out != "X"
    print("\n4. Agregate")
    print(f"  D pe cadrele evaluate: ML {np.mean(labels_out[graded] == 'D'):.1%} | V5 {np.mean(v5[graded] == 'D'):.1%}"
          f" | DFL, ca reper: 40-52%")
    print(f"  dovadă de minge (<= {VOTE_DIST_M:.0f} m) pe {np.mean(pd.notna(evidence[~dead])):.1%} din cadrele în joc")
    describe("V5", v5)
    describe("ML", labels_out)

    print()
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(out_csv)], check=False)


if __name__ == "__main__":
    main()
