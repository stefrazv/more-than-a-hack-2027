"""H/A on degraded DFL matches: PathCRF, the ball (rule-based V5) and combinations.

All variants use the same D (the model trained on degraded+stitched players of the other
6 matches) and are scored on J03WR9 with the official metric.

H/A methods (only on frames predicted "alive"):
  PathCRF          PathCRF alone
  ball (V5)        the V5 automaton: team of the player nearest to the ball, if within
                   <= d metres and >= 0.75 m ahead of the opponent; a change is confirmed after
                   25 frames; possession restarts after every stoppage; gaps are filled
  ball > PathCRF   per frame: the ball if it is clearly with one team, otherwise PathCRF
  PathCRF+vote     PathCRF possessions are kept, but a possession changes team if the ball is
                   clearly with the other team on >= 60% of its evidence frames (at least 1 s)
Each with d in {1, 2} m and minimum possession tau in {1, 2, 3, 4} s.

The DFL ball is real while play is alive (it only freezes at stoppages), and degrade.py
removes 41% of it, as in the competition data. The submission uses PathCRF+vote, d = 2 m,
tau = 2 s. Also provides ball_evidence() and vote(), used by run_ml.py.

    python src/ha_fusion_dfl.py --dfl <folder with the preprocessed DFL matches (*.parquet)>
"""
from pathlib import Path
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dead_dfl import frame_features  # noqa: E402
from src.eval_dfl import ground_truth, load_model, min_duration, pathcrf_labels, score, with_half_gaps  # noqa: E402
from src.robust_dfl import TEST, degraded_table, load, predict_dead, train, with_episodes  # noqa: E402

PATHCRF = ROOT / "third_party" / "pathcrf"

MARGIN_M = 0.75
CONFIRM = 25
TAUS = (1, 2, 3, 4)
DISTS = (1.0, 2.0)
VOTE_SHARE, VOTE_MIN = 0.6, 25


def ball_evidence(table, max_dist):
    """Per frame: 'H'/'A' if the ball is clearly with one team, otherwise None."""
    bx, by = table["ball_x"].to_numpy(float), table["ball_y"].to_numpy(float)
    out = np.full(len(table), None, dtype=object)
    near = {}
    for side in ("home", "away"):
        xs = table[[c for c in table.columns if c.startswith(f"{side}_") and c.endswith("_x") and c.count("_") == 2]].to_numpy(float)
        ys = table[[c[:-2] + "_y" for c in table.columns if c.startswith(f"{side}_") and c.endswith("_x") and c.count("_") == 2]].to_numpy(float)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # frames without ball or players -> NaN
            near[side] = np.nanmin(np.hypot(xs - bx[:, None], ys - by[:, None]), axis=1)
    h, a = near["home"], near["away"]
    with np.errstate(invalid="ignore"):
        out[(h <= max_dist) & (a - h >= MARGIN_M)] = "H"
        out[(a <= max_dist) & (h - a >= MARGIN_M)] = "A"
    return out


def v5_automaton(evidence, dead):
    """The V5 H/A automaton (hysteresis), reset at every stoppage; gaps filled backwards, then forwards."""
    labels = np.full(len(evidence), None, dtype=object)
    current, pending, count, start = None, None, 0, 0
    for t, cand in enumerate(evidence):
        if dead[t]:
            current, pending, count = None, None, 0
            continue
        if current is None:
            current = cand
            labels[t] = current
            continue
        if cand is None or cand == current:
            pending, count = None, 0
        else:
            if cand != pending:
                pending, count, start = cand, 1, t
            else:
                count += 1
            if count >= CONFIRM:
                current = pending
                labels[start:t + 1] = current
                pending, count = None, 0
        labels[t] = current if labels[t] is None else labels[t]
    s = pd.Series(labels)
    return s.bfill().ffill().to_numpy(dtype=object)


def vote(pathcrf, evidence, dead):
    """A PathCRF possession changes team if the ball clearly contradicts it."""
    out = pathcrf.copy()
    n, i = len(out), 0
    while i < n:
        j = i
        while j < n and out[j] == out[i] and dead[j] == dead[i]:
            j += 1
        if not dead[i] and out[i] in ("H", "A"):
            ev = evidence[i:j]
            ev = ev[pd.notna(ev)]
            if len(ev) >= VOTE_MIN:
                other = "A" if out[i] == "H" else "H"
                if np.mean(ev == other) >= VOTE_SHARE:
                    out[i:j] = other
        i = j
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl", required=True, help="folder with the preprocessed DFL matches (*.parquet)")
    parser.add_argument("--seed", type=int, default=47)
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

    print("Degrading and stitching the 7 DFL matches...")
    raw = {m: load(folder, m) for m in matches}
    dead_true = {m: ground_truth(raw[m])[0] == "D" for m in matches}
    tables = {m: degraded_table(raw[m], opts.seed, calculate_running_features) for m in matches}
    feats = {m: frame_features(tables[m], False) for m in matches}

    print("D model on degraded players...")
    model_d = train([feats[m] for m in train_matches], [dead_true[m] for m in train_matches])
    table = tables[TEST]
    period = table["period_id"].to_numpy()
    dead = predict_dead(model_d, feats[TEST], period)
    gt, _ = ground_truth(raw[TEST])
    gt_full = with_half_gaps(gt, period)

    print("PathCRF...")
    model_pc, args = load_model(PATHCRF, device)
    pathcrf, _ = pathcrf_labels(model_pc, args, with_episodes(table, dead))
    pathcrf = pd.Series(pathcrf).ffill().bfill().to_numpy(dtype=object)

    methods = {"PathCRF": pathcrf}
    for d in DISTS:
        ev = ball_evidence(table, d)
        print(f"  ball evidence at <= {d:.0f} m: {np.mean(pd.notna(ev[~dead])):.1%} of alive frames")
        methods[f"ball (V5), {d:.0f} m"] = v5_automaton(ev, dead)
        methods[f"ball > PathCRF, {d:.0f} m"] = np.where(pd.notna(ev), ev, pathcrf)
        methods[f"PathCRF+vote, {d:.0f} m"] = vote(pathcrf, ev, dead)

    print(f"\n{TEST}, seed {opts.seed}, D from the model on degraded players")
    print(f"{'H/A method':<26}{'tau':>5}{'spellF1':>9}{'macroF1':>9}{'spells':>9}{'true':>7}{'team':>8}")
    best = None
    for name, team in methods.items():
        for tau in TAUS:
            pred = np.where(dead, "D", team)
            pred = np.where(pd.isna(pred), "D", pred).astype(object)
            s, m, n_pred, n_gt, _, _, tm = score(gt_full, with_half_gaps(min_duration(pred, tau), period))
            print(f"{name:<26}{tau:>4}s{s:>9.4f}{m:>9.4f}{n_pred:>9}{n_gt:>7}{tm:>8.4f}")
            if best is None or s > best[0]:
                best = (s, name, tau)
        print()
    print(f"Best: {best[1]}, tau = {best[2]} s, spellF1 {best[0]:.4f}")


if __name__ == "__main__":
    main()
