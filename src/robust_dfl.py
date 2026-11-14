"""The ML pipeline on DFL matches DEGRADED to look like the competition tracking.

For each DFL match: players -> degrade.py (anonymous slots, gaps, duplicates, noise)
-> stitch.py (back to 11 players per team) -> speeds (PathCRF's function) -> models.
Test match J03WR9 (unseen by PathCRF); the D model is trained on the other 6 matches,
in two versions: on CLEAN players and on DEGRADED+STITCHED players.

Rows:
  clean:    D model + PathCRF                   -> reference (0.604)
  degraded: clean D model + PathCRF             -> a model trained on clean data
  degraded: degraded D model + PathCRF          -> what the submission uses
  degraded: perfect D + PathCRF                 -> loss due to H/A alone
  degraded: degraded D model + perfect team     -> loss due to D alone

Also provides the helpers used by run_ml.py and train_dead_model.py.

    python src/robust_dfl.py --dfl <folder with the preprocessed DFL matches (*.parquet)>
    (optional: --seeds 47 48 49 to average over several degradations; takes longer)
"""
from pathlib import Path
import argparse
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dead_dfl import SEED, SUBSAMPLE, frame_features, smooth  # noqa: E402
from src.degrade import degrade  # noqa: E402
from src.eval_dfl import ground_truth, load_model, min_duration, pathcrf_labels, score, with_half_gaps  # noqa: E402
from src.stitch import stitch  # noqa: E402

PATHCRF = ROOT / "third_party" / "pathcrf"

TEST = "J03WR9"
MIN_DEAD_S, MIN_ALIVE_S = 4, 2
TAU_S = 1


def load(folder, match):
    return pd.read_parquet(folder / f"{match}.parquet").sort_values(["period_id", "frame_id"]).reset_index(drop=True)


def degraded_table(df, seed, calculate_running_features):
    """The DFL match's players degraded and stitched back, in PathCRF format (11 per team)."""
    names = sorted(c[:-2] for c in df.columns if re.fullmatch(r"(home|away)_[^_]+_x", c))
    team = np.array([0 if n.startswith("home") else 1 for n in names])
    parts = []
    for period in sorted(df["period_id"].unique()):
        sub = df[df["period_id"] == period]
        players = np.stack([sub[[f"{n}_x" for n in names]].to_numpy(float),
                            sub[[f"{n}_y" for n in names]].to_numpy(float)], axis=-1)
        ball = sub[["ball_x", "ball_y"]].to_numpy(float)
        slots, slot_team, ball_out = degrade(players, team, ball, seed=seed + int(period))
        tracks, _ = stitch(slots, slot_team, 0, len(sub) - 1)

        table = pd.DataFrame({
            "frame_id": sub["frame_id"].to_numpy(),
            "period_id": period,
            "timestamp": sub["timestamp"].to_numpy(),
            "phase_id": period,
            "episode_id": 1,
            "ball_state": "alive",
            "ball_owning_team_id": np.nan,
            "ball_x": ball_out[:, 0],     # degraded ball (41% missing); the D model and PathCRF do not use it
            "ball_y": ball_out[:, 1],
        })
        cols = {}
        for k, side in ((0, "home"), (1, "away")):
            for j in range(tracks[k].shape[1]):
                cols[f"{side}_{j + 1}_x"] = tracks[k][:, j, 0]
                cols[f"{side}_{j + 1}_y"] = tracks[k][:, j, 1]
        parts.append(pd.concat([table, pd.DataFrame(cols)], axis=1))
    return calculate_running_features(pd.concat(parts, ignore_index=True)).reset_index(drop=True)


def train(feature_list, label_list):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
    model.fit(pd.concat([f.iloc[::SUBSAMPLE] for f in feature_list]),
              np.concatenate([y[::SUBSAMPLE] for y in label_list]))
    return model


def predict_dead(model, feats, period):
    return smooth(model.predict_proba(feats)[:, 1] > 0.5, period, MIN_DEAD_S, MIN_ALIVE_S)


def with_episodes(table, dead):
    """PathCRF episodes of play, from D (predicted or true)."""
    t = table.copy()
    period = t["period_id"].to_numpy()
    alive = ~dead
    starts = alive & ~np.r_[False, alive[:-1]] | (alive & np.r_[True, period[1:] != period[:-1]])
    t["episode_id"] = np.where(alive, np.cumsum(starts), 0)
    t["ball_state"] = np.where(alive, "alive", "dead")
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl", required=True, help="folder with the preprocessed DFL matches (*.parquet)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[47])
    parser.add_argument("--cpu", action="store_true")
    opts = parser.parse_args()
    sys.path.insert(0, str(PATHCRF))
    import torch
    from datatools.utils import calculate_running_features
    pd.set_option("future.no_silent_downcasting", True)
    device = "cpu" if opts.cpu or not torch.cuda.is_available() else "cuda:0"

    folder = Path(opts.dfl)
    matches = sorted(p.stem for p in folder.glob("*.parquet"))
    train_matches = [m for m in matches if m != TEST]

    print("Loading the DFL matches...")
    raw = {m: load(folder, m) for m in matches}
    gt = {m: ground_truth(raw[m])[0] for m in matches}
    dead_true = {m: gt[m] == "D" for m in matches}
    clean_feats = {m: frame_features(raw[m], False) for m in matches}

    test_df = raw[TEST]
    period = test_df["period_id"].to_numpy()
    gt_full = with_half_gaps(gt[TEST], period)
    true_team = pd.Series(np.where(dead_true[TEST], None, gt[TEST])).ffill().bfill().to_numpy(dtype=object)
    model_pc, args = load_model(PATHCRF, device)

    def evaluate(dead, team):
        pred = np.where(dead, "D", team)
        pred = np.where(pd.isna(pred), "D", pred).astype(object)
        return score(gt_full, with_half_gaps(min_duration(pred, TAU_S), period))

    print("D model on clean players (the other 6 matches)...")
    model_clean = train([clean_feats[m] for m in train_matches], [dead_true[m] for m in train_matches])

    rows = {}
    # reference on clean data
    dead = predict_dead(model_clean, clean_feats[TEST], period)
    team, _ = pathcrf_labels(model_pc, args, with_episodes(test_df, dead))
    rows["clean:    D model + PathCRF"] = [evaluate(dead, team)]

    for seed in opts.seeds:
        start = time.time()
        print(f"\n--- seed {seed}: degrading and stitching the 7 matches ---")
        tables = {m: degraded_table(raw[m], seed, calculate_running_features) for m in matches}
        deg_feats = {m: frame_features(tables[m], False) for m in matches}
        model_deg = train([deg_feats[m] for m in train_matches], [dead_true[m] for m in train_matches])
        test_table = tables[TEST]

        variants = {
            "degraded: clean D model + PathCRF": predict_dead(model_clean, deg_feats[TEST], period),
            "degraded: degraded D model + PathCRF": predict_dead(model_deg, deg_feats[TEST], period),
            "degraded: perfect D + PathCRF": dead_true[TEST],
        }
        for name, dead in variants.items():
            team, _ = pathcrf_labels(model_pc, args, with_episodes(test_table, dead))
            rows.setdefault(name, []).append(evaluate(dead, team))
        dead = variants["degraded: degraded D model + PathCRF"]
        rows.setdefault("degraded: degraded D model + perfect team", []).append(evaluate(dead, true_team))
        print(f"seed {seed}: {time.time() - start:.0f} s")

    print(f"\nTest match {TEST}, seeds {opts.seeds} (mean)")
    print(f"{'variant':<44}{'spellF1':>8}{'macroF1':>9}{'spells':>9}{'true':>7}{'alive/dead F1':>14}{'team':>8}")
    for name, results in rows.items():
        r = np.array(results, dtype=float)
        s, m, n_pred, n_gt, _, ad, tm = r.mean(axis=0)
        print(f"{name:<44}{s:>8.4f}{m:>9.4f}{n_pred:>9.0f}{n_gt:>7.0f}{ad:>14.4f}{tm:>8.4f}")
    print("\nCompetition baseline (different match, for orientation only): spellF1 0.2433")


if __name__ == "__main__":
    main()
