"""D smoothing on degraded DFL matches: the current rule vs a two-state HMM (Viterbi).

Same setup as ha_fusion_dfl.py: the 7 DFL matches are degraded and stitched (seed --seed),
the D model is trained on the 6 matches other than J03WR9, and everything is scored on
J03WR9 with the official metric. Only the step that turns the D model's probabilities into
dead / alive changes:

  rule 4 s / 2 s        what the submission uses: threshold 0.5, then drop stoppages < 4 s
                        and restarts < 2 s (dead_dfl.smooth)
  HMM w=.. L x..        dead_dfl.viterbi_dead: mean dead / alive lengths measured on the
                        6 training matches (times L), emission weight w

H/A is then the submission's own: PathCRF + ball vote at 2 m, minimum possession 2 s.

    python src/hmm_dfl.py --dfl <folder with the preprocessed DFL matches (*.parquet)> --seed 47
"""
from pathlib import Path
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dead_dfl import FPS, frame_features, mean_run_lengths, smooth, viterbi_dead  # noqa: E402
from src.eval_dfl import ground_truth, load_model, min_duration, pathcrf_labels, score, with_half_gaps  # noqa: E402
from src.ha_fusion_dfl import ball_evidence, vote  # noqa: E402
from src.robust_dfl import TEST, degraded_table, load, train, with_episodes  # noqa: E402

PATHCRF = ROOT / "third_party" / "pathcrf"
MIN_DEAD_S, MIN_ALIVE_S = 4, 2
TAU_S = 2
VOTE_DIST_M = 2.0
WEIGHTS = (1.0, 0.2, 0.05, 0.02)
SCALES = (0.5, 1.0, 2.0)


def n_stoppages(dead, period):
    return int(sum(np.sum(np.diff(np.r_[0, dead[period == p].astype(np.int8)]) == 1) for p in np.unique(period)))


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
    warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
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
    p_dead = model_d.predict_proba(feats[TEST])[:, 1]
    gt, _ = ground_truth(raw[TEST])
    gt_full = with_half_gaps(gt, period)

    lengths = [mean_run_lengths(dead_true[m], raw[m]["period_id"].to_numpy()) for m in train_matches]
    mean_dead = float(np.mean([d for d, _ in lengths]))
    mean_alive = float(np.mean([a for _, a in lengths]))
    print(f"  measured on the training matches: mean stoppage {mean_dead / FPS:.1f} s, "
          f"mean live stretch {mean_alive / FPS:.1f} s")

    variants = {f"rule {MIN_DEAD_S} s / {MIN_ALIVE_S} s": smooth(p_dead > 0.5, period, MIN_DEAD_S, MIN_ALIVE_S)}
    for w in WEIGHTS:
        for k in SCALES:
            variants[f"HMM w={w:g} L x{k:g}"] = viterbi_dead(p_dead, period, k * mean_dead, k * mean_alive, w)

    print(f"PathCRF for {len(variants)} D variants...")
    model_pc, args = load_model(PATHCRF, device)
    evidence = ball_evidence(table, VOTE_DIST_M)
    true_dead = gt == "D"

    print(f"\n{TEST}, seed {opts.seed}; H/A = PathCRF + vote {VOTE_DIST_M:g} m, tau {TAU_S} s")
    print(f"  true D {true_dead.mean():.1%}, true stoppages {n_stoppages(true_dead, period)}")
    print(f"{'D smoothing':<22}{'D':>7}{'stops':>7}{'spellF1':>9}{'macroF1':>9}{'alive/dead F1':>15}{'team':>8}")
    best = None
    for name, dead in variants.items():
        team, _ = pathcrf_labels(model_pc, args, with_episodes(table, dead))
        team = pd.Series(team).ffill().bfill().to_numpy(dtype=object)
        team = vote(team, evidence, dead)
        pred = np.where(dead, "D", team)
        pred = np.where(pd.isna(pred), "D", pred).astype(object)
        s, m, _, _, _, ad, tm = score(gt_full, with_half_gaps(min_duration(pred, TAU_S), period))
        print(f"{name:<22}{dead.mean():>7.1%}{n_stoppages(dead, period):>7}{s:>9.4f}{m:>9.4f}{ad:>15.4f}{tm:>8.4f}",
              flush=True)
        if best is None or s > best[0]:
            best = (s, name)
    print(f"Best: {best[1]}, spellF1 {best[0]:.4f}")


if __name__ == "__main__":
    main()
