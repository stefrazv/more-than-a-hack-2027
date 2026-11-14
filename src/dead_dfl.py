"""Dead-ball (D) detector, trained on the DFL matches (external data).

Per-frame inputs, computable on the competition tracking too:
  players -- mean/median/p90 speed, share standing still, spread and centre of each team;
  ball    -- visible, outside the pitch, its speed, distance to the nearest player;
  plus means over centred windows of 1, 3 and 6 s and the "after minus before" trend.
Model: HistGradientBoostingClassifier (scikit-learn), then smoothing (stoppages and
restarts shorter than a few seconds are removed).

Evaluation: leave-one-match-out over the 7 matches. To isolate D, the H/A team is taken
from the DFL truth and D from the detector; scored with the official metric (spellF1).

Three configurations:
  players          -- no ball (robust to a missing ball)
  players+ball     -- clean DFL ball
  players+ball41   -- ball removed on 41% of frames (gaps ~2 s), as in the competition data

Result: players+ball reaches ~0.999 frame accuracy because the DFL ball freezes exactly
when play stops (see check_dfl_ball.py) -- an artefact the competition data does not have.
The submission therefore uses the player-only features (frame_features(df, False)).

    python src/dead_dfl.py --dfl <folder with the preprocessed DFL matches (*.parquet)>
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
from src.eval_dfl import ground_truth, score, with_half_gaps  # noqa: E402

FPS = 25
PITCH_X, PITCH_Y = 105.0, 68.0
SLOW = 1.0                      # m/s: below this speed a player is "standing still"
WINDOWS_S = (1, 3, 6)
SUBSAMPLE = 5                   # train on one frame in 5 (5 fps)
MIN_RUNS_S = (0, 1, 2, 4)       # grid for the smoothing
BALL_MISSING = 0.41
SEED = 47


# ------------------------------------------------------------------ inputs

def frame_features(df, use_ball, ball_mask=None):
    xcols = sorted(c for c in df.columns if re.fullmatch(r"(home|away)_[^_]+_x", c))
    names = [c[:-2] for c in xcols]
    X = df[[f"{p}_x" for p in names]].to_numpy(float)
    Y = df[[f"{p}_y" for p in names]].to_numpy(float)
    S = df[[f"{p}_speed" for p in names]].to_numpy(float)
    home = np.array([p.startswith("home") for p in names])

    f = {}
    with np.errstate(all="ignore"):
        f["speed_mean"] = np.nanmean(S, 1)
        f["speed_med"] = np.nanmedian(S, 1)
        f["speed_p90"] = np.nanpercentile(S, 90, axis=1)
        f["slow_share"] = np.nansum(S < SLOW, 1) / np.maximum(np.isfinite(S).sum(1), 1)
        f["n_players"] = np.isfinite(X).sum(1)
        for tag, m in (("h", home), ("a", ~home)):
            f[f"cx_{tag}"] = np.abs(np.nanmean(X[:, m], 1) - PITCH_X / 2)
            f[f"sx_{tag}"] = np.nanstd(X[:, m], 1)
            f[f"sy_{tag}"] = np.nanstd(Y[:, m], 1)
            f[f"speed_{tag}"] = np.nanmean(S[:, m], 1)
        f["centroid_gap"] = np.hypot(np.nanmean(X[:, home], 1) - np.nanmean(X[:, ~home], 1),
                                     np.nanmean(Y[:, home], 1) - np.nanmean(Y[:, ~home], 1))

        if use_ball:
            bx, by = df["ball_x"].to_numpy(float).copy(), df["ball_y"].to_numpy(float).copy()
            if ball_mask is not None:
                bx[ball_mask] = np.nan
                by[ball_mask] = np.nan
            seen = np.isfinite(bx)
            f["ball_seen"] = seen.astype(float)
            f["ball_out"] = (seen & ((bx < 0) | (bx > PITCH_X) | (by < 0) | (by > PITCH_Y))).astype(float)
            f["ball_speed"] = np.r_[np.nan, np.hypot(np.diff(bx), np.diff(by)) * FPS]
            f["ball_near"] = np.nanmin(np.hypot(X - bx[:, None], Y - by[:, None]), 1)
            f["ball_cx"] = np.abs(bx - PITCH_X / 2)

    feats = pd.DataFrame(f, index=df.index)
    period = df["period_id"].to_numpy()
    rolled = {}
    base = ["speed_mean", "slow_share"] + (["ball_seen", "ball_speed", "ball_near", "ball_out"] if use_ball else [])
    for col in base:
        g = feats[col].groupby(period)
        for w in WINDOWS_S:
            n = w * FPS
            rolled[f"{col}_c{w}"] = g.transform(lambda s: s.rolling(n, center=True, min_periods=1).mean())
        past = g.transform(lambda s: s.rolling(2 * FPS, min_periods=1).mean())
        future = g.transform(lambda s: s[::-1].rolling(2 * FPS, min_periods=1).mean()[::-1])
        rolled[f"{col}_trend"] = future - past
    return pd.concat([feats, pd.DataFrame(rolled, index=df.index)], axis=1).astype(np.float32)


def ball_gaps(n, period, rng):
    """Ball mask: exponential gaps (mean 2 s, max 30 s) until 41% of each half is missing."""
    mask = np.zeros(n, dtype=bool)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        target = int(BALL_MISSING * len(idx))
        while mask[idx].sum() < target:
            length = min(int(rng.exponential(2 * FPS)) + 1, 30 * FPS)
            start = rng.integers(0, len(idx))
            mask[idx[start:start + length]] = True
    return mask


# ------------------------------------------------------------------ smoothing

def drop_short_runs(flags, period, min_len, value):
    """Runs of `value` shorter than min_len frames (within a half) are flipped."""
    out = flags.copy()
    if min_len <= 0:
        return out
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        f = out[idx]
        edges = np.flatnonzero(np.diff(np.r_[~f[0], f, ~f[-1]].astype(np.int8)))
        for a, b in zip(edges[:-1], edges[1:]):
            if f[a] == value and b - a < min_len:
                f[a:b] = not value
        out[idx] = f
    return out


def smooth(dead, period, min_dead_s, min_alive_s):
    """Drop stoppages shorter than min_dead_s, then live stretches shorter than min_alive_s."""
    dead = drop_short_runs(dead, period, int(min_dead_s * FPS), True)
    return drop_short_runs(dead, period, int(min_alive_s * FPS), False)


def mean_run_lengths(dead, period):
    """Mean length in frames of the dead runs and of the alive runs (within halves)."""
    runs = {True: [], False: []}
    for p in np.unique(period):
        f = dead[period == p]
        edges = np.flatnonzero(np.diff(np.r_[~f[0], f, ~f[-1]].astype(np.int8)))
        for a, b in zip(edges[:-1], edges[1:]):
            runs[bool(f[a])].append(b - a)
    return float(np.mean(runs[True])), float(np.mean(runs[False]))


def viterbi_dead(p_dead, period, mean_dead, mean_alive, weight=1.0):
    """Most likely dead/alive sequence under a two-state HMM, per half.

    States: 0 = alive, 1 = dead. Staying in a state has probability 1 - 1/L, with L its mean
    length in frames (mean_alive, mean_dead); leaving it 1/L. The emission score of a frame
    is weight * log p (dead) or weight * log(1 - p) (alive), p being the classifier's
    probability. weight < 1 discounts the evidence, since consecutive frames are strongly
    correlated and are not independent observations.
    """
    p = np.clip(np.asarray(p_dead, dtype=float), 1e-6, 1 - 1e-6)
    stay = np.log1p(-1.0 / np.array([mean_alive, mean_dead]))
    leave = np.log(1.0 / np.array([mean_alive, mean_dead]))
    out = np.zeros(len(p), dtype=bool)
    for half in np.unique(period):
        idx = np.flatnonzero(period == half)
        e_alive = weight * np.log1p(-p[idx])
        e_dead = weight * np.log(p[idx])
        n = len(idx)
        came_from_other = np.zeros((n, 2), dtype=bool)   # True: best predecessor is the other state
        s_alive, s_dead = e_alive[0], e_dead[0]
        for t in range(1, n):
            a_stay, a_switch = s_alive + stay[0], s_dead + leave[1]
            d_stay, d_switch = s_dead + stay[1], s_alive + leave[0]
            came_from_other[t, 0] = a_switch > a_stay
            came_from_other[t, 1] = d_switch > d_stay
            s_alive = max(a_stay, a_switch) + e_alive[t]
            s_dead = max(d_stay, d_switch) + e_dead[t]
        state = 1 if s_dead > s_alive else 0
        path = np.empty(n, dtype=bool)
        for t in range(n - 1, -1, -1):
            path[t] = bool(state)
            if came_from_other[t, state]:
                state = 1 - state
        out[idx] = path
    return out


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl", required=True, help="folder with the preprocessed DFL matches (*.parquet)")
    opts = parser.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    folder = Path(opts.dfl)
    matches = sorted(p.stem for p in folder.glob("*.parquet"))
    rng = np.random.default_rng(SEED)

    print("Loading the matches and computing the inputs...")
    data = {}
    for m in matches:
        df = pd.read_parquet(folder / f"{m}.parquet").sort_values(["period_id", "frame_id"]).reset_index(drop=True)
        gt, _ = ground_truth(df)
        period = df["period_id"].to_numpy()
        mask = ball_gaps(len(df), period, rng)
        data[m] = dict(
            gt=gt, period=period, dead=(gt == "D"),
            feats={"players": frame_features(df, False),
                   "players+ball": frame_features(df, True),
                   "players+ball41": frame_features(df, True, mask)},
        )
        print(f"  {m}: {len(df)} frames, true D {np.mean(gt == 'D'):.1%}")

    grid = [(d, a) for d in MIN_RUNS_S for a in MIN_RUNS_S]
    for config in ("players", "players+ball", "players+ball41"):
        start = time.time()
        results = {g: [] for g in grid}
        frame_acc, d_share = [], []
        for test in matches:
            train = [m for m in matches if m != test]
            Xtr = pd.concat([data[m]["feats"][config].iloc[::SUBSAMPLE] for m in train])
            ytr = np.concatenate([data[m]["dead"][::SUBSAMPLE] for m in train])
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
            model.fit(Xtr, ytr)

            t = data[test]
            raw = model.predict_proba(t["feats"][config])[:, 1] > 0.5
            frame_acc.append(np.mean(raw == t["dead"]))
            team = np.where(t["dead"], None, t["gt"])
            team = pd.Series(team).ffill().bfill().to_numpy(dtype=object)   # the true team, also on frames predicted "alive"
            gt_full = with_half_gaps(t["gt"], t["period"])
            for g in grid:
                dead = smooth(raw, t["period"], *g)
                pred = with_half_gaps(np.where(dead, "D", team).astype(object), t["period"])
                results[g].append(score(gt_full, pred))
            d_share.append((np.mean(raw), np.mean(t["dead"])))

        print(f"\n=== {config}  ({time.time() - start:.0f} s) ===")
        ds = np.array(d_share)
        print(f"  frame accuracy (no smoothing): {np.mean(frame_acc):.4f} | predicted D {ds[:, 0].mean():.1%} vs true {ds[:, 1].mean():.1%}")
        print(f"  {'smoothing (dead, alive)':<26}{'spellF1':>9}{'macroF1':>9}{'spells':>9}{'true':>7}{'alive/dead F1':>14}")
        best = max(grid, key=lambda g: np.mean([r[0] for r in results[g]]))
        for g in [(0, 0), (1, 1), (2, 2), (4, 4), best]:
            r = np.array(results[g], dtype=float)
            mark = "  <- best" if g == best else ""
            print(f"  {f'min {g[0]} s, min {g[1]} s':<26}{r[:, 0].mean():>9.4f}{r[:, 1].mean():>9.4f}"
                  f"{r[:, 2].mean():>9.0f}{r[:, 3].mean():>7.0f}{r[:, 5].mean():>14.4f}{mark}")
        print(f"  spellF1 per match (best): " + " ".join(f"{m}={r[0]:.3f}" for m, r in zip(matches, results[best])))


if __name__ == "__main__":
    main()
