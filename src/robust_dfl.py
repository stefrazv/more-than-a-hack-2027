"""T-02: soluția ML pe meciuri DFL STRICATE ca tracking-ul nostru.

Pe fiecare meci DFL: jucători -> degrade.py (sloturi anonime, pauze, dubluri, zgomot)
-> stitch.py (înapoi la 11 jucători pe echipă) -> vitezele (funcția PathCRF) -> modelele.
Testăm pe J03WR9 (nevăzut de PathCRF); modelul de D se antrenează pe celelalte 6 meciuri,
în două variante: pe jucători CURAȚI (ca acum) și pe jucători STRICAȚI+LIPIȚI (T-12).

Rânduri:
  curat:   D model + PathCRF                 -> referința (0,604)
  stricat: D model curat + PathCRF           -> ce am avea azi pe meciul nostru
  stricat: D model stricat + PathCRF         -> varianta T-12
  stricat: D perfect + PathCRF               -> cât pierde doar H/A
  stricat: D model stricat + echipă perfectă -> cât pierde doar D

    E:\\mtah\\pathcrf\\.venv\\Scripts\\python.exe src\\robust_dfl.py --pathcrf E:\\mtah\\pathcrf
    (opțional: --seeds 47 48 49 pentru media pe mai multe stricări; durează mai mult)
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

TEST = "J03WR9"
MIN_DEAD_S, MIN_ALIVE_S = 4, 2
TAU_S = 1


def load(folder, match):
    return pd.read_parquet(folder / f"{match}.parquet").sort_values(["period_id", "frame_id"]).reset_index(drop=True)


def degraded_table(df, seed, calculate_running_features):
    """Jucătorii meciului DFL stricați și lipiți la loc, în formatul PathCRF (11 pe echipă)."""
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
            "ball_x": ball_out[:, 0],     # mingea stricată (41% lipsă); modelele de D/PathCRF n-o folosesc
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
    """Episoadele de joc pentru PathCRF, din D (prezis sau adevărat)."""
    t = table.copy()
    period = t["period_id"].to_numpy()
    alive = ~dead
    starts = alive & ~np.r_[False, alive[:-1]] | (alive & np.r_[True, period[1:] != period[:-1]])
    t["episode_id"] = np.where(alive, np.cumsum(starts), 0)
    t["ball_state"] = np.where(alive, "alive", "dead")
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathcrf", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[47])
    parser.add_argument("--cpu", action="store_true")
    opts = parser.parse_args()
    sys.path.insert(0, str(opts.pathcrf))
    import torch
    from datatools.utils import calculate_running_features
    pd.set_option("future.no_silent_downcasting", True)
    device = "cpu" if opts.cpu or not torch.cuda.is_available() else "cuda:0"

    folder = Path(opts.pathcrf) / "data" / "sportec" / "tracking_processed"
    matches = sorted(p.stem for p in folder.glob("*.parquet"))
    train_matches = [m for m in matches if m != TEST]

    print("Încarc meciurile DFL...")
    raw = {m: load(folder, m) for m in matches}
    gt = {m: ground_truth(raw[m])[0] for m in matches}
    dead_true = {m: gt[m] == "D" for m in matches}
    clean_feats = {m: frame_features(raw[m], False) for m in matches}

    test_df = raw[TEST]
    period = test_df["period_id"].to_numpy()
    gt_full = with_half_gaps(gt[TEST], period)
    true_team = pd.Series(np.where(dead_true[TEST], None, gt[TEST])).ffill().bfill().to_numpy(dtype=object)
    model_pc, args = load_model(opts.pathcrf, device)

    def evaluate(dead, team):
        pred = np.where(dead, "D", team)
        pred = np.where(pd.isna(pred), "D", pred).astype(object)
        return score(gt_full, with_half_gaps(min_duration(pred, TAU_S), period))

    print("Modelul de D pe jucători curați (celelalte 6 meciuri)...")
    model_clean = train([clean_feats[m] for m in train_matches], [dead_true[m] for m in train_matches])

    rows = {}
    # referința pe date curate
    dead = predict_dead(model_clean, clean_feats[TEST], period)
    team, _ = pathcrf_labels(model_pc, args, with_episodes(test_df, dead))
    rows["curat:   D model + PathCRF"] = [evaluate(dead, team)]

    for seed in opts.seeds:
        start = time.time()
        print(f"\n--- seed {seed}: stric și lipesc cele 7 meciuri ---")
        tables = {m: degraded_table(raw[m], seed, calculate_running_features) for m in matches}
        deg_feats = {m: frame_features(tables[m], False) for m in matches}
        model_deg = train([deg_feats[m] for m in train_matches], [dead_true[m] for m in train_matches])
        test_table = tables[TEST]

        variants = {
            "stricat: D model curat + PathCRF": predict_dead(model_clean, deg_feats[TEST], period),
            "stricat: D model stricat + PathCRF": predict_dead(model_deg, deg_feats[TEST], period),
            "stricat: D perfect + PathCRF": dead_true[TEST],
        }
        for name, dead in variants.items():
            team, _ = pathcrf_labels(model_pc, args, with_episodes(test_table, dead))
            rows.setdefault(name, []).append(evaluate(dead, team))
        dead = variants["stricat: D model stricat + PathCRF"]
        rows.setdefault("stricat: D model stricat + echipă perfectă", []).append(evaluate(dead, true_team))
        print(f"seed {seed}: {time.time() - start:.0f} s")

    print(f"\nMeciul de test {TEST}, seed-uri {opts.seeds} (media)")
    print(f"{'varianta':<44}{'spellF1':>8}{'macroF1':>9}{'posesii':>9}{'adev.':>7}{'joc/oprit F1':>14}{'echipă':>8}")
    for name, results in rows.items():
        r = np.array(results, dtype=float)
        s, m, n_pred, n_gt, _, ad, tm = r.mean(axis=0)
        print(f"{name:<44}{s:>8.4f}{m:>9.4f}{n_pred:>9.0f}{n_gt:>7.0f}{ad:>14.4f}{tm:>8.4f}")
    print("\nBaseline-ul concursului (alt meci, orientativ): spellF1 0.2433")


if __name__ == "__main__":
    main()
