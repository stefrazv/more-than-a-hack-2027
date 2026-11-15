"""Dead-ball model trained on one degradation per match vs. on several (I2).

Same setup as hmm_dfl.py. For each test seed s, J03WR9 is degraded with seed s and scored with
the official metric, using the submission's own pipeline after the D model: HMM decoding
(w = 0.2, mean lengths measured on the 6 training matches), PathCRF + ball vote at 2 m,
minimum possession 2 s. Only the D model's training data changes:

  A  the 6 training matches degraded once, with seed s         (what the submission does)
  B  the 6 training matches degraded with every seed in --seeds (3x more, more varied data)

Check: A on seed 47 must give 0.4279, the "HMM w=0.2 L x1" row of hmm_dfl.py.

    python src/multiseed_dfl.py --dfl <folder with the preprocessed DFL matches (*.parquet)>
"""
from pathlib import Path
import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dead_dfl import frame_features, mean_run_lengths, viterbi_dead  # noqa: E402
from src.eval_dfl import ground_truth, load_model, min_duration, pathcrf_labels, score, with_half_gaps  # noqa: E402
from src.ha_fusion_dfl import ball_evidence, vote  # noqa: E402
from src.robust_dfl import TEST, degraded_table, load, train, with_episodes  # noqa: E402

PATHCRF = ROOT / "third_party" / "pathcrf"
HMM_WEIGHT = 0.2
TAU_S = 2
VOTE_DIST_M = 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl", required=True, help="folder with the preprocessed DFL matches (*.parquet)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[47, 48, 49])
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

    raw = {m: load(folder, m) for m in matches}
    dead_true = {m: ground_truth(raw[m])[0] == "D" for m in matches}
    gt, _ = ground_truth(raw[TEST])
    lengths = [mean_run_lengths(dead_true[m], raw[m]["period_id"].to_numpy()) for m in train_matches]
    mean_dead = float(np.mean([d for d, _ in lengths]))
    mean_alive = float(np.mean([a for _, a in lengths]))

    # Features of every training match for every seed; the full table only for the test match.
    feats, test_tables = {}, {}
    for seed in opts.seeds:
        start = time.time()
        print(f"Degrading and stitching the 7 DFL matches, seed {seed}...", flush=True)
        feats[seed] = {}
        for m in matches:
            table = degraded_table(raw[m], seed, calculate_running_features)
            feats[seed][m] = frame_features(table, False)
            if m == TEST:
                test_tables[seed] = table
            del table
        print(f"  seed {seed}: {time.time() - start:.0f} s", flush=True)

    print("Training the D models...", flush=True)
    models_a = {s: train([feats[s][m] for m in train_matches], [dead_true[m] for m in train_matches])
                for s in opts.seeds}
    model_b = train([feats[s][m] for s in opts.seeds for m in train_matches],
                    [dead_true[m] for s in opts.seeds for m in train_matches])

    model_pc, args = load_model(PATHCRF, device)

    def evaluate(model, seed):
        table = test_tables[seed]
        period = table["period_id"].to_numpy()
        p_dead = model.predict_proba(feats[seed][TEST])[:, 1]
        dead = viterbi_dead(p_dead, period, mean_dead, mean_alive, HMM_WEIGHT)
        team, _ = pathcrf_labels(model_pc, args, with_episodes(table, dead))
        team = pd.Series(team).ffill().bfill().to_numpy(dtype=object)
        team = vote(team, ball_evidence(table, VOTE_DIST_M), dead)
        pred = np.where(dead, "D", team)
        pred = np.where(pd.isna(pred), "D", pred).astype(object)
        s, m, _, _, _, ad, tm = score(with_half_gaps(gt, period), with_half_gaps(min_duration(pred, TAU_S), period))
        return s, ad, dead.mean()

    rows = []
    for seed in opts.seeds:
        print(f"PathCRF, test seed {seed}...", flush=True)
        a, b = evaluate(models_a[seed], seed), evaluate(model_b, seed)
        rows.append((seed, a, b))

    print(f"\n{TEST}; D = HMM w={HMM_WEIGHT:g}; H/A = PathCRF + vote {VOTE_DIST_M:g} m, tau {TAU_S} s")
    print(f"  A = D model on one degradation per match, B = on seeds {opts.seeds}")
    print(f"{'test seed':<10}{'A spellF1':>11}{'B spellF1':>11}{'B - A':>9}{'A D':>8}{'B D':>8}{'A al/dead F1':>14}{'B al/dead F1':>14}")
    for seed, a, b in rows:
        print(f"{seed:<10}{a[0]:>11.4f}{b[0]:>11.4f}{b[0] - a[0]:>+9.4f}{a[2]:>8.1%}{b[2]:>8.1%}{a[1]:>14.4f}{b[1]:>14.4f}")
    mean_a = np.mean([a[0] for _, a, _ in rows])
    mean_b = np.mean([b[0] for _, _, b in rows])
    print(f"{'mean':<10}{mean_a:>11.4f}{mean_b:>11.4f}{mean_b - mean_a:>+9.4f}")
    print("\nB goes into the submission only if B - A >= +0.02 on average.")


if __name__ == "__main__":
    main()
