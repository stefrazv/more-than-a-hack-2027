"""PathCRF on a DFL match, scored with the competition's official metric.

Every DFL frame carries the ball status (ball_state) and the team in possession
(ball_owning_team_id) -- the same kind of labels as the competition ground truth. We turn
them into H/A/D and score our variants with evaluation/possession_metrics.py, exactly as
the organisers do. Also provides the shared helpers (labels, minimum possession, PathCRF
loading and inference) used by the other scripts and by run_ml.py.

Default match: J03WR9, the only DFL match the pretrained model saved/140 did not see in training.

Variants:
  E1  D = DFL truth ("oracle"), H/A = PathCRF        -> isolates the H/A quality
  E2  no D information at all, H/A = PathCRF everywhere
  E3  as E2, but frames where PathCRF sees the ball go out become D
Each with a minimum possession tau (a shorter change of team is ignored).

    python src/eval_dfl.py --dfl <folder with the preprocessed DFL matches (*.parquet)>
"""
from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATHCRF = ROOT / "third_party" / "pathcrf"
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
TRIAL_ID = 140
TAUS = (0, 1, 2, 3, 4, 6, 8)   # seconds


# ------------------------------------------------------------------ labels

def ground_truth(df):
    """H/A/D from the DFL columns. Home team = the team owning the ball when player_id is home_*."""
    alive = df["ball_state"].astype(str).str.lower().str.contains("alive").to_numpy()
    owner = df["ball_owning_team_id"].astype("string")

    pid = df["player_id"].astype("string") if "player_id" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    home_ids = owner[pid.str.startswith("home", na=False)].mode()
    if home_ids.empty:
        raise ValueError("cannot find the home team id")
    home_id = home_ids.iloc[0]

    is_home = owner.eq(home_id).fillna(False).to_numpy(dtype=bool)
    has_owner = owner.notna().to_numpy(dtype=bool)
    team = np.where(is_home, "H", np.where(has_owner, "A", None)).astype(object)
    team = pd.Series(team).groupby(df["period_id"].to_numpy()).transform(lambda s: s.ffill().bfill()).to_numpy()
    return np.where(alive, team, "D").astype(object), home_id


def with_half_gaps(labels, period):
    """Insert one X frame between halves, so the scorer does not join possessions across half-time."""
    first = np.r_[False, period[1:] != period[:-1]]
    return np.insert(labels, np.flatnonzero(first), "X")


def min_duration(labels, tau_s):
    """An H/A possession shorter than tau that directly follows the other team's possession goes back to that team."""
    if tau_s <= 0:
        return labels
    out = labels.copy()
    min_len = int(round(tau_s * FPS))
    i, n = 0, len(out)
    prev_team = None
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        lab = out[i]
        if lab in ("H", "A"):
            if prev_team is not None and lab != prev_team and j - i < min_len:
                out[i:j] = prev_team
                lab = prev_team
            prev_team = lab
        else:
            prev_team = None          # D or X: after a stoppage, possession starts fresh
        i = j
    return out


def score(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"], r["mean_spell_duration_s"], r["alive_dead_f1"], r["team_acc_both_alive"]


# ------------------------------------------------------------------ PathCRF

def load_model(pathcrf_dir, device):
    import torch
    from models.utils import build_model, load_trial_args, resolve_model_path
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    save_path = str(Path(pathcrf_dir) / "saved" / f"{TRIAL_ID:03d}")
    args = load_trial_args(save_path)
    model = build_model(args, device=device)
    model.load_state_dict(torch.load(resolve_model_path(save_path, "state_dict_best_acc.pt"),
                                     map_location=device, weights_only=False))
    return model, args


def pathcrf_labels(model, args, df):
    """Run PathCRF; returns (H/A team per frame, ball-out flag per frame), aligned with df."""
    from inference import inference
    _, _, micro, _ = inference(model, df, use_crf=args.get("crf_weight", 0) > 0, decode="indep",
                               window_seconds=args.get("window_seconds", None),
                               correct_episode_lasts=False, evaluate=False)
    micro = micro.reindex(df["frame_id"].to_numpy())
    src = micro["edge_src"].astype("string")
    dst = micro["edge_dst"].astype("string")
    side = src.str.split("_").str[0]
    out = (side.eq("out") | dst.str.startswith("out", na=False)).fillna(False).to_numpy(dtype=bool)
    team = side.map({"home": "H", "away": "A"}).astype(object)
    team = pd.Series(team.to_numpy()).groupby(df["episode_id"].to_numpy()).transform(lambda s: s.ffill().bfill())
    return team.to_numpy(dtype=object), out


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl", required=True, help="folder with the preprocessed DFL matches (*.parquet)")
    parser.add_argument("--match", default="J03WR9")
    parser.add_argument("--cpu", action="store_true")
    opts = parser.parse_args()

    sys.path.insert(0, str(PATHCRF))
    import torch
    pd.set_option("future.no_silent_downcasting", True)
    device = "cpu" if opts.cpu or not torch.cuda.is_available() else "cuda:0"

    path = Path(opts.dfl) / f"{opts.match}.parquet"
    df = pd.read_parquet(path).sort_values(["period_id", "frame_id"]).reset_index(drop=True)
    assert df["frame_id"].is_unique, "frame_id is not unique"
    period = df["period_id"].to_numpy()

    gt, home_id = ground_truth(df)
    gt_full = with_half_gaps(gt, period)
    print(f"Match {opts.match}: {len(df)} frames | D {np.mean(gt == 'D'):.1%} | home {np.mean(gt[gt != 'D'] == 'H'):.1%} (id {home_id})")

    model, args = load_model(PATHCRF, device)

    # E1: episodes = the stretches where the ball is alive according to DFL
    start = time.time()
    team1, _ = pathcrf_labels(model, args, df)
    print(f"E1 inference: {time.time() - start:.0f} s")
    print(f"   coverage of alive frames: {np.mean(pd.notna(team1[gt != 'D'])):.1%}")
    e1 = np.where(gt == "D", "D", team1)

    # E2/E3: no D information -- a single "alive" episode per half
    blind = df.copy()
    blind["ball_state"] = "alive"
    blind["episode_id"] = blind["period_id"]
    players = [c for c in blind.columns if c.startswith(("home_", "away_"))]
    blind[players] = blind.groupby("phase_id")[players].transform(lambda s: s.ffill().bfill())
    start = time.time()
    team2, out2 = pathcrf_labels(model, args, blind)
    print(f"E2 inference: {time.time() - start:.0f} s")
    print(f"   coverage: {np.mean(pd.notna(team2)):.1%}")
    e2 = team2.copy()
    e3 = np.where(out2, "D", team2)
    print(f"Ball out according to PathCRF (E3): {out2.mean():.1%} of frames; true D: {np.mean(gt == 'D'):.1%}")

    print(f"\n{'variant':<34}{'spellF1':>8}{'macroF1':>9}{'spells':>9}{'true':>7}{'dur(s)':>8}{'alive/dead F1':>14}{'team':>8}")
    for name, pred in (("E1 oracle D + PathCRF", e1), ("E2 no D", e2), ("E3 D from PathCRF 'out'", e3)):
        for tau in TAUS:
            p = with_half_gaps(min_duration(np.where(pd.isna(pred), "D", pred).astype(object), tau), period)
            s, m, n_pred, n_gt, dur, ad, team = score(gt_full, p)
            print(f"{name + f'  tau={tau}s':<34}{s:>8.4f}{m:>9.4f}{n_pred:>9}{n_gt:>7}{dur:>8.1f}{ad:>14.4f}{team:>8.4f}")
        print()


if __name__ == "__main__":
    main()
