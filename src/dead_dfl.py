"""Detector de minge oprită (D), antrenat pe meciurile DFL (date externe).

Intrări pe cadru, calculabile și pe tracking-ul nostru:
  jucători -- viteza medie/mediană/p90, ce parte stă pe loc, răsfirarea și centrul echipelor;
  minge    -- e vizibilă, e în afara terenului, viteza ei, distanța la cel mai apropiat jucător;
  plus medii pe ferestre centrate de 1, 3 și 6 s și diferența "după minus înainte".
Model: HistGradientBoostingClassifier (scikit-learn), apoi netezire (opriri și reluări
mai scurte de câteva secunde sunt eliminate).

Evaluare: leave-one-match-out pe cele 7 meciuri. Ca să izolăm D, echipa H/A e luată din
adevărul DFL, iar D din detector; notăm cu metrica oficială (spellF1).

Trei configurații:
  jucători          -- fără minge (robust la mingea lipsă)
  jucători+minge    -- mingea DFL curată
  jucători+minge41  -- mingea ștearsă pe 41% din cadre (goluri ~2 s), ca la noi, provizoriu
                       până e gata degrade.py

Se rulează din repo-ul nostru, cu Python-ul mediului PathCRF:
    E:\\mtah\\pathcrf\\.venv\\Scripts\\python.exe src\\dead_dfl.py --pathcrf E:\\mtah\\pathcrf
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
SLOW = 1.0                      # m/s: sub această viteză un jucător "stă pe loc"
WINDOWS_S = (1, 3, 6)
SUBSAMPLE = 5                   # antrenăm pe un cadru din 5 (5 fps)
MIN_RUNS_S = (0, 1, 2, 4)       # grila pentru netezire
BALL_MISSING = 0.41
SEED = 47


# ------------------------------------------------------------------ intrări

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
    """Mască provizorie: goluri exponențiale (media 2 s, max 30 s) până la 41% din fiecare repriză."""
    mask = np.zeros(n, dtype=bool)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        target = int(BALL_MISSING * len(idx))
        while mask[idx].sum() < target:
            length = min(int(rng.exponential(2 * FPS)) + 1, 30 * FPS)
            start = rng.integers(0, len(idx))
            mask[idx[start:start + length]] = True
    return mask


# ------------------------------------------------------------------ netezire

def drop_short_runs(flags, period, min_len, value):
    """Secvențele de `value` mai scurte de min_len cadre (în interiorul unei reprize) se inversează."""
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
    dead = drop_short_runs(dead, period, int(min_dead_s * FPS), True)
    return drop_short_runs(dead, period, int(min_alive_s * FPS), False)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathcrf", required=True)
    opts = parser.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    folder = Path(opts.pathcrf) / "data" / "sportec" / "tracking_processed"
    matches = sorted(p.stem for p in folder.glob("*.parquet"))
    rng = np.random.default_rng(SEED)

    print("Încarc meciurile și calculez intrările...")
    data = {}
    for m in matches:
        df = pd.read_parquet(folder / f"{m}.parquet").sort_values(["period_id", "frame_id"]).reset_index(drop=True)
        gt, _ = ground_truth(df)
        period = df["period_id"].to_numpy()
        mask = ball_gaps(len(df), period, rng)
        data[m] = dict(
            gt=gt, period=period, dead=(gt == "D"),
            feats={"jucători": frame_features(df, False),
                   "jucători+minge": frame_features(df, True),
                   "jucători+minge41": frame_features(df, True, mask)},
        )
        print(f"  {m}: {len(df)} cadre, D real {np.mean(gt == 'D'):.1%}")

    grid = [(d, a) for d in MIN_RUNS_S for a in MIN_RUNS_S]
    for config in ("jucători", "jucători+minge", "jucători+minge41"):
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
            team = pd.Series(team).ffill().bfill().to_numpy(dtype=object)   # echipa adevărată și pe cadrele prezise "în joc"
            gt_full = with_half_gaps(t["gt"], t["period"])
            for g in grid:
                dead = smooth(raw, t["period"], *g)
                pred = with_half_gaps(np.where(dead, "D", team).astype(object), t["period"])
                results[g].append(score(gt_full, pred))
            d_share.append((np.mean(raw), np.mean(t["dead"])))

        print(f"\n=== {config}  ({time.time() - start:.0f} s) ===")
        ds = np.array(d_share)
        print(f"  acuratețe pe cadru (fără netezire): {np.mean(frame_acc):.4f} | D prezis {ds[:, 0].mean():.1%} vs real {ds[:, 1].mean():.1%}")
        print(f"  {'netezire (oprire, joc)':<26}{'spellF1':>9}{'macroF1':>9}{'posesii':>9}{'adev.':>7}{'joc/oprit F1':>14}")
        best = max(grid, key=lambda g: np.mean([r[0] for r in results[g]]))
        for g in [(0, 0), (1, 1), (2, 2), (4, 4), best]:
            r = np.array(results[g], dtype=float)
            mark = "  <- cel mai bun" if g == best else ""
            print(f"  {f'min {g[0]} s, min {g[1]} s':<26}{r[:, 0].mean():>9.4f}{r[:, 1].mean():>9.4f}"
                  f"{r[:, 2].mean():>9.0f}{r[:, 3].mean():>7.0f}{r[:, 5].mean():>14.4f}{mark}")
        print(f"  spellF1 pe meci (cel mai bun): " + " ".join(f"{m}={r[0]:.3f}" for m, r in zip(matches, results[best])))


if __name__ == "__main__":
    main()
