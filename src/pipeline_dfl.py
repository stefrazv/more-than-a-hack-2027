"""Soluția întreagă pe un meci DFL: D din modelul pe jucători + H/A din PathCRF.

1. modelul de D (doar jucători) se antrenează pe celelalte 6 meciuri DFL;
2. pe meciul de test prezice "oprit / în joc" și netezește (opriri >= 4 s, reluări >= 2 s);
3. PathCRF rulează doar pe porțiunile prezise "în joc" și dă H/A (posesie minimă 1 s);
4. notare cu metrica oficială, față de adevărul DFL.

Pentru comparație mai afișează două rânduri de referință, cu câte o componentă perfectă.

    E:\\mtah\\pathcrf\\.venv\\Scripts\\python.exe src\\pipeline_dfl.py --pathcrf E:\\mtah\\pathcrf
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dead_dfl import SEED, SUBSAMPLE, frame_features, smooth  # noqa: E402
from src.eval_dfl import ground_truth, load_model, min_duration, pathcrf_labels, score, with_half_gaps  # noqa: E402

TEST = "J03WR9"
MIN_DEAD_S, MIN_ALIVE_S = 4, 2     # netezirea D, aleasă pe DFL (dead_dfl.py)
TAU_S = 1                          # posesia minimă, aleasă pe DFL (eval_dfl.py)


def load(folder, match):
    return pd.read_parquet(folder / f"{match}.parquet").sort_values(["period_id", "frame_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathcrf", required=True)
    parser.add_argument("--cpu", action="store_true")
    opts = parser.parse_args()
    sys.path.insert(0, str(opts.pathcrf))
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    pd.set_option("future.no_silent_downcasting", True)
    device = "cpu" if opts.cpu or not torch.cuda.is_available() else "cuda:0"

    folder = Path(opts.pathcrf) / "data" / "sportec" / "tracking_processed"
    matches = sorted(p.stem for p in folder.glob("*.parquet"))

    # 1. modelul de D, antrenat fără meciul de test
    print("1. Antrenez modelul de D pe celelalte 6 meciuri...")
    X, y = [], []
    for m in matches:
        if m == TEST:
            continue
        df = load(folder, m)
        gt, _ = ground_truth(df)
        X.append(frame_features(df, False).iloc[::SUBSAMPLE])
        y.append((gt == "D")[::SUBSAMPLE])
    model_d = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
    model_d.fit(pd.concat(X), np.concatenate(y))

    # 2. D prezis pe meciul de test
    df = load(folder, TEST)
    gt, _ = ground_truth(df)
    period = df["period_id"].to_numpy()
    dead = model_d.predict_proba(frame_features(df, False))[:, 1] > 0.5
    dead = smooth(dead, period, MIN_DEAD_S, MIN_ALIVE_S)
    print(f"2. {TEST}: D prezis {dead.mean():.1%}, D real {np.mean(gt == 'D'):.1%}")

    # 3. PathCRF doar pe porțiunile prezise "în joc"
    pred_df = df.copy()
    alive = ~dead
    starts = alive & ~np.r_[False, alive[:-1]] | (alive & np.r_[True, period[1:] != period[:-1]])
    pred_df["episode_id"] = np.where(alive, np.cumsum(starts), 0)
    pred_df["ball_state"] = np.where(alive, "alive", "dead")
    print("3. PathCRF pe porțiunile în joc...")
    model, args = load_model(opts.pathcrf, device)
    team, _ = pathcrf_labels(model, args, pred_df)

    # PathCRF și pe porțiunile în joc reale, pentru rândul de referință
    team_true_d, _ = pathcrf_labels(model, args, df)

    # 4. notare
    gt_full = with_half_gaps(gt, period)
    true_team = pd.Series(np.where(gt == "D", None, gt)).ffill().bfill().to_numpy(dtype=object)

    rows = {
        "SOLUȚIA: D model + PathCRF": np.where(dead, "D", team),
        "ref: D perfect + PathCRF": np.where(gt == "D", "D", team_true_d),
        "ref: D model + echipă perfectă": np.where(dead, "D", true_team),
    }
    print(f"\n{'varianta':<34}{'spellF1':>8}{'macroF1':>9}{'posesii':>9}{'adev.':>7}{'joc/oprit F1':>14}{'echipă':>8}")
    for name, pred in rows.items():
        pred = np.where(pd.isna(pred), "D", pred).astype(object)
        s, m, n_pred, n_gt, _, ad, tm = score(gt_full, with_half_gaps(min_duration(pred, TAU_S), period))
        print(f"{name:<34}{s:>8.4f}{m:>9.4f}{n_pred:>9}{n_gt:>7}{ad:>14.4f}{tm:>8.4f}")
    print("\nBaseline-ul concursului (alt meci, doar orientativ): spellF1 0.2433")


if __name__ == "__main__":
    main()
