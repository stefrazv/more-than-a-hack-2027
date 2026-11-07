from pathlib import Path
import csv
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_activity as ca

ROOT = ca.ROOT
DEAD = ROOT / "dead_ball_candidates.csv"

FPS = ca.FPS
MIN_GAP = ca.LONG_GAP_S * FPS   # gol de minge > 10 s
FILL_HOLES = 2 * FPS            # umple pauzele < 2 s intre episoade
MIN_EPISODE = 5 * FPS           # elimina episoadele < 5 s


def true_runs(mask):
    """Segmente maximale de True: lista de (start, end) inclusiv."""
    runs = []
    start = None

    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(mask) - 1))

    return runs


def long_ball_gaps(valid, min_gap=MIN_GAP):
    gap = np.zeros(len(valid), dtype=bool)

    for start, end in ca.PLAY_INTERVALS:
        run_start = None

        for frame in range(start, end + 2):
            missing = frame <= end and not valid[frame]

            if missing and run_start is None:
                run_start = frame

            if not missing and run_start is not None:
                if frame - run_start > min_gap:
                    gap[run_start:frame] = True
                run_start = None

    return gap


def fill_holes(mask, max_len):
    out = mask.copy()
    runs = true_runs(mask)

    for (_, end_a), (start_b, _) in zip(runs, runs[1:]):
        if start_b - end_a - 1 < max_len:
            out[end_a + 1:start_b] = True

    return out


def drop_short(mask, min_len):
    out = mask.copy()

    for start, end in true_runs(mask):
        if end - start + 1 < min_len:
            out[start:end + 1] = False

    return out


def read_dead(n):
    with DEAD.open(newline="", encoding="utf-8") as file:
        dead = np.array(
            [row["dead_candidate"] == "1" for row in csv.DictReader(file)]
        )

    assert len(dead) == n
    return dead


def summary(name, mask, graded_total):
    runs = true_runs(mask)
    frames = int(mask.sum())
    median_s = (
        np.median([e - s + 1 for s, e in runs]) / FPS if runs else 0.0
    )

    print(
        f"{name:<26} | {frames:>7} | {100 * frames / graded_total:>5.1f}% | "
        f"{len(runs):>8} | {median_s:>10.1f}"
    )

def sensitivity(speed, valid, graded, pause_speed, graded_total):
    print("\nvarianta                   |   cadre |  % eval | episoade | mediana (s)")
    print("---------------------------+---------+---------+----------+------------")

    for gap_s in (5, 10, 15, 20):
        for pct in (50, 75):
            ref = np.percentile(pause_speed, pct)
            slow = np.isfinite(speed) & (speed <= ref)
            raw = long_ball_gaps(valid, gap_s * FPS) & slow
            mask = drop_short(fill_holes(raw, FILL_HOLES), MIN_EPISODE) & graded
            summary(f"gol>{gap_s}s, viteza<=p{pct}", mask, graded_total)

def main():
    xs, ys = ca.read_players()
    speed = ca.median_player_speed(xs, ys)
    valid = ca.read_ball_valid()
    n = len(valid)

    graded = np.zeros(n, dtype=bool)
    for start, end in ca.EVALUATED_INTERVALS:
        graded[start:end + 1] = True
    graded_total = int(graded.sum())

    pause = np.zeros(n, dtype=bool)
    pause[ca.OFFICIAL_PAUSE[0]:ca.OFFICIAL_PAUSE[1] + 1] = True
    pause_speed = speed[pause & np.isfinite(speed)]
    reference = np.percentile(pause_speed, 75)

    slow = np.isfinite(speed) & (speed <= reference)

    raw = long_ball_gaps(valid) & slow
    new_rule = drop_short(fill_holes(raw, FILL_HOLES), MIN_EPISODE) & graded

    current = read_dead(n) & graded
    union = current | new_rule

    print(f"Referinta de viteza (p75 pauza): {reference:.2f} m/s")
    print(f"Cadre evaluate: {graded_total}\n")
    print("varianta                   |   cadre |  % eval | episoade | mediana (s)")
    print("---------------------------+---------+---------+----------+------------")

    summary("D actual (V5)", current, graded_total)
    summary("regula noua", new_rule, graded_total)
    summary("uniune (actual + noua)", union, graded_total)

    added = new_rule & ~current
    # print(f"\nCadre D aduse doar de regula noua: {int(added.sum())}")
    # print(
    #     "Din regula noua, deja D in V5: "
    #     f"{int((new_rule & current).sum())} cadre"
    # )

    sensitivity(speed, valid, graded, pause_speed, graded_total)


if __name__ == "__main__":
    main()