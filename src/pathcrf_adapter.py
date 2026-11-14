"""Turn the competition ATD into PathCRF's input table, and summarise possession labels.

build_tracking(): ATD slots -> 11 continuous players per team (src/stitch.py) -> one row per
frame in PathCRF's format (columns home_k_x, away_k_y, ...), over the two halves.
describe(): aggregate statistics of a label sequence (number and length of possessions).

Used by run_ml.py. An earlier version of this file also combined PathCRF with the rule-based
V5 labels; that experiment is superseded by run_ml.py.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.check_slots import FPS, GRADED
from src.paths import OUT_DIR
from src.stitch import stitch

PITCH_X_OURS = 105.3      # games.py
PITCH_X_PATHCRF = 105.0   # PathCRF's datatools/config.py
OUT = OUT_DIR / "pathcrf_out"


def build_tracking(slots_xy, slot_team, labels):
    """PathCRF input table, one row per frame, halves only.

    Frames whose label is H or A form the episodes of live play; run_ml.py passes all "H"
    here and sets the real episodes later from the dead-ball model (robust_dfl.with_episodes).
    """
    parts = []
    last_episode = 0
    for period, (a, b) in enumerate(GRADED, start=1):
        tracks, stats = stitch(slots_xy, slot_team, a, b)
        for team, name in ((0, "home"), (1, "away")):
            print(f"  half {period}, {name}: observed {100 * stats[team]['observed_share']:.1f}%, "
                  f"forced joins {stats[team]['forced']}")

        frames = np.arange(a, b + 1)
        alive = np.isin(labels[a:b + 1], ["H", "A"])
        starts = alive & ~np.r_[False, alive[:-1]]
        episode = np.cumsum(starts) + last_episode
        if alive.any():
            last_episode = int(episode[alive].max())

        df = pd.DataFrame({
            "frame_id": frames,
            "period_id": period,
            "timestamp": (frames - a) / FPS,
            "phase_id": period,
            "episode_id": np.where(alive, episode, 0),
            "ball_state": np.where(alive, "alive", "dead"),
            "ball_owning_team_id": np.nan,
            "ball_x": np.nan,
            "ball_y": np.nan,
        })
        cols = {}
        for team, name in ((0, "home"), (1, "away")):
            for k in range(tracks[team].shape[1]):
                cols[f"{name}_{k + 1}_x"] = tracks[team][:, k, 0] * PITCH_X_PATHCRF / PITCH_X_OURS
                cols[f"{name}_{k + 1}_y"] = tracks[team][:, k, 1]
        parts.append(pd.concat([df, pd.DataFrame(cols)], axis=1))

    return pd.concat(parts, ignore_index=True)


def spells(labels):
    """Possessions = maximal runs of H or of A on the graded frames (D interrupts them)."""
    lengths = []
    for a, b in GRADED:
        prev, run = None, 0
        for lab in labels[a:b + 1]:
            if lab in ("H", "A") and lab == prev:
                run += 1
            else:
                if prev in ("H", "A"):
                    lengths.append(run)
                prev, run = lab, 1
        if prev in ("H", "A"):
            lengths.append(run)
    return np.array(lengths)


def describe(name, labels):
    s = spells(labels)
    ha = np.isin(labels, ["H", "A"])
    home = np.mean(labels[ha] == "H")
    print(f"  {name:<10} possessions {len(s):>4} | mean length {s.mean() / FPS:5.1f} s | "
          f"possessions <= 1 s: {100 * np.mean(s <= FPS):4.1f}% | Barcelona share {home:.3f}")
