"""The whole pipeline on a CLEAN DFL match: D from the player-only model + H/A from PathCRF.

1. the D model (players only) is trained on the other 6 DFL matches;
2. on the test match it predicts dead / alive and smooths (stoppages >= 4 s, restarts >= 2 s);
3. PathCRF runs only on the stretches predicted alive and gives H/A (minimum possession 1 s);
4. scored with the official metric against the DFL truth.

For comparison it also prints two reference rows, each with one perfect component.

    python src/pipeline_dfl.py --dfl <folder with the preprocessed DFL matches (*.parquet)>
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

PATHCRF = ROOT / "third_party" / "pathcrf"

TEST = "J03WR9"
MIN_DEAD_S, MIN_ALIVE_S = 4, 2     # D smoothing, chosen on DFL (dead_dfl.py)
TAU_S = 1                          # minimum possession, chosen on DFL (eval_dfl.py)


def load(folder, match):
    return pd.read_parquet(folder / f"{match}.parquet").sort_values(["period_id", "frame_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl", required=True, help="folder with the preprocessed DFL matches (*.parquet)")
    parser.add_argument("--cpu", action="store_true")
    opts = parser.parse_args()
    sys.path.insert(0, str(PATHCRF))
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    pd.set_option("future.no_silent_downcasting", True)
    device = "cpu" if opts.cpu or not torch.cuda.is_available() else "cuda:0"

    folder = Path(opts.dfl)
    matches = sorted(p.stem for p in folder.glob("*.parquet"))

    # 1. the D model, trained without the test match
    print("1. Training the D model on the other 6 matches...")
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

    # 2. predicted D on the test match
    df = load(folder, TEST)
    gt, _ = ground_truth(df)
    period = df["period_id"].to_numpy()
    dead = model_d.predict_proba(frame_features(df, False))[:, 1] > 0.5
    dead = smooth(dead, period, MIN_DEAD_S, MIN_ALIVE_S)
    print(f"2. {TEST}: predicted D {dead.mean():.1%}, true D {np.mean(gt == 'D'):.1%}")

    # 3. PathCRF only on the stretches predicted alive
    pred_df = df.copy()
    alive = ~dead
    starts = alive & ~np.r_[False, alive[:-1]] | (alive & np.r_[True, period[1:] != period[:-1]])
    pred_df["episode_id"] = np.where(alive, np.cumsum(starts), 0)
    pred_df["ball_state"] = np.where(alive, "alive", "dead")
    print("3. PathCRF on the alive stretches...")
    model, args = load_model(PATHCRF, device)
    team, _ = pathcrf_labels(model, args, pred_df)

    # PathCRF on the true alive stretches too, for the reference row
    team_true_d, _ = pathcrf_labels(model, args, df)

    # 4. scoring
    gt_full = with_half_gaps(gt, period)
    true_team = pd.Series(np.where(gt == "D", None, gt)).ffill().bfill().to_numpy(dtype=object)

    rows = {
        "PIPELINE: D model + PathCRF": np.where(dead, "D", team),
        "ref: perfect D + PathCRF": np.where(gt == "D", "D", team_true_d),
        "ref: D model + perfect team": np.where(dead, "D", true_team),
    }
    print(f"\n{'variant':<34}{'spellF1':>8}{'macroF1':>9}{'spells':>9}{'true':>7}{'alive/dead F1':>14}{'team':>8}")
    for name, pred in rows.items():
        pred = np.where(pd.isna(pred), "D", pred).astype(object)
        s, m, n_pred, n_gt, _, ad, tm = score(gt_full, with_half_gaps(min_duration(pred, TAU_S), period))
        print(f"{name:<34}{s:>8.4f}{m:>9.4f}{n_pred:>9}{n_gt:>7}{ad:>14.4f}{tm:>8.4f}")
    print("\nCompetition baseline (different match, for orientation only): spellF1 0.2433")


if __name__ == "__main__":
    main()
