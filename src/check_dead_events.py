from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_activity as ca
import check_dead_activity as cd
from starter.load_smart_tagging import load

FPS = ca.FPS
RADIUS = 10 * FPS
TRIALS = 300
rng = np.random.default_rng(0)


def episodes_for(speed, valid, graded, pause_speed, gap_s, pct):
    ref = np.percentile(pause_speed, pct)
    slow = np.isfinite(speed) & (speed <= ref)
    raw = cd.long_ball_gaps(valid, gap_s * FPS) & slow
    mask = cd.drop_short(cd.fill_holes(raw, cd.FILL_HOLES), cd.MIN_EPISODE) & graded
    return cd.true_runs(mask)


def near_share(episodes, windows):
    hits = 0
    for s, e in episodes:
        if any(ws - RADIUS <= e and we + RADIUS >= s for ws, we in windows):
            hits += 1
    return hits / len(episodes)


def main():
    xs, ys = ca.read_players()
    speed = ca.median_player_speed(xs, ys)
    valid = ca.read_ball_valid()
    n = len(valid)

    graded = np.zeros(n, dtype=bool)
    for a, b in ca.EVALUATED_INTERVALS:
        graded[a:b + 1] = True
    starts = np.flatnonzero(graded)

    pause = np.zeros(n, dtype=bool)
    pause[ca.OFFICIAL_PAUSE[0]:ca.OFFICIAL_PAUSE[1] + 1] = True
    pause_speed = speed[pause & np.isfinite(speed)]

    windows = [
        (e["start_frame"], e["end_frame"])
        for e in load()
        if str(e.get("code", "")).strip().upper() == "SET PIECES"
    ]
    print("Ferestre SET PIECES:", len(windows))
    print(f"Raza de cautare: {RADIUS // FPS} s\n")
    print("varianta               | episoade | real | aleator (medie) | aleator p95")
    print("-----------------------+----------+------+-----------------+------------")

    for gap_s, pct in [(10, 75), (10, 50), (15, 50)]:
        eps = episodes_for(speed, valid, graded, pause_speed, gap_s, pct)
        lengths = [e - s + 1 for s, e in eps]
        real = near_share(eps, windows)

        nulls = []
        for _ in range(TRIALS):
            rnd = []
            for length in lengths:
                s = int(rng.choice(starts))
                rnd.append((s, s + length - 1))
            nulls.append(near_share(rnd, windows))

        print(
            f"gol>{gap_s}s, <=p{pct:<10} | {len(eps):>8} | {real:>4.0%} | "
            f"{np.mean(nulls):>15.0%} | {np.percentile(nulls, 95):>10.0%}"
        )


if __name__ == "__main__":
    main()