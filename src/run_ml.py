"""Soluția ML pe meciul nostru: D din modelul pe jucători + H/A din PathCRF.

1. modelul de D se antrenează pe toate cele 7 meciuri DFL (date externe) și se salvează;
2. pe meciul nostru: sloturile se lipesc în 11 jucători pe echipă (stitch.py),
   se calculează vitezele, modelul prezice "oprit / în joc", apoi netezire (4 s / 2 s);
3. PathCRF rulează pe porțiunile prezise "în joc" și dă H/A; posesie minimă 1 s;
4. în afara reprizelor X, iar înainte de lovitura de început a reprizei a doua D (ca în V5).

Toți parametrii au fost aleși pe DFL. Afișează doar agregate.
Fișiere: pathcrf_out/submission_ml.csv și pathcrf_out/dead_model.joblib (ignorate de git).

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
from src.check_slots import GRADED, load_slots  # noqa: E402
from src.dead_dfl import SEED, SUBSAMPLE, frame_features, smooth  # noqa: E402
from src.eval_dfl import ground_truth, load_model, min_duration, pathcrf_labels  # noqa: E402
from src.paths import ATD_PATH, FEATURES, META_PATH, SUBMISSION  # noqa: E402
from src.pathcrf_adapter import OUT, build_tracking, describe  # noqa: E402

N_FRAMES = 199_319
SECOND_HALF_EVAL_START, SECOND_HALF_KICKOFF = 110808, 113742   # DATA.md / FAQ.md
MIN_DEAD_S, MIN_ALIVE_S = 4, 2
TAU_S = 1


def train_dead_model(pathcrf_dir):
    from sklearn.ensemble import HistGradientBoostingClassifier
    folder = Path(pathcrf_dir) / "data" / "sportec" / "tracking_processed"
    X, y = [], []
    for path in sorted(folder.glob("*.parquet")):
        df = pd.read_parquet(path).sort_values(["period_id", "frame_id"]).reset_index(drop=True)
        gt, _ = ground_truth(df)
        X.append(frame_features(df, False).iloc[::SUBSAMPLE])
        y.append((gt == "D")[::SUBSAMPLE])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
    model.fit(pd.concat(X), np.concatenate(y))
    return model


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

    print("1. Antrenez modelul de D pe cele 7 meciuri DFL...")
    model_d = train_dead_model(opts.pathcrf)
    joblib.dump(model_d, OUT / "dead_model.joblib")

    print("2. Meciul nostru: lipirea sloturilor și vitezele...")
    slots_xy, slot_team = load_slots(ATD_PATH, META_PATH)
    all_alive = np.full(N_FRAMES, "H", dtype=object)
    table = calculate_running_features(build_tracking(slots_xy, slot_team, all_alive)).reset_index(drop=True)
    period = table["period_id"].to_numpy()
    frames = table["frame_id"].to_numpy()

    dead = model_d.predict_proba(frame_features(table, False))[:, 1] > 0.5
    dead = smooth(dead, period, MIN_DEAD_S, MIN_ALIVE_S)

    print("3. PathCRF pe porțiunile prezise în joc...")
    alive = ~dead
    starts = alive & ~np.r_[False, alive[:-1]] | (alive & np.r_[True, period[1:] != period[:-1]])
    table["episode_id"] = np.where(alive, np.cumsum(starts), 0)
    table["ball_state"] = np.where(alive, "alive", "dead")
    model, args = load_model(opts.pathcrf, device)
    team, _ = pathcrf_labels(model, args, table)

    # 4. fișierul complet
    v5 = pd.read_csv(SUBMISSION)["label"].to_numpy(dtype=object)
    labels = np.full(N_FRAMES, "X", dtype=object)
    labels[SECOND_HALF_EVAL_START:SECOND_HALF_KICKOFF] = "D"
    fallback = np.where(np.isin(v5[frames], ["H", "A"]), v5[frames], "D")
    labels[frames] = np.where(dead, "D", np.where(pd.isna(team), fallback, team))
    labels = min_duration(labels, TAU_S)
    out_csv = OUT / "submission_ml.csv"
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(out_csv, index=False)

    # agregate
    graded = labels != "X"
    print("\n4. Agregate")
    print(f"  D pe cadrele evaluate: ML {np.mean(labels[graded] == 'D'):.1%} | V5 {np.mean(v5[graded] == 'D'):.1%}"
          f" | DFL, ca reper: 40-52%")
    print(f"  cadre în joc fără etichetă PathCRF (completate din V5): {int(np.sum(alive & pd.isna(team)))}")
    feat = pd.read_csv(FEATURES)
    home_d, away_d = feat["nearest_home_m"].to_numpy(), feat["nearest_away_m"].to_numpy()
    clear_h = (home_d <= 2.0) & (away_d - home_d >= 0.75)
    clear_a = (away_d <= 2.0) & (home_d - away_d >= 0.75)
    clear = np.isin(labels, ["H", "A"]) & (clear_h | clear_a)
    print(f"  acord cu mingea vizibilă și clar la o echipă: {np.mean(labels[clear] == np.where(clear_h, 'H', 'A')[clear]):.3f}")
    describe("V5", v5)
    describe("ML", labels)

    print()
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(out_csv)], check=False)


if __name__ == "__main__":
    main()
