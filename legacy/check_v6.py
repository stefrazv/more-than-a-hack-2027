from pathlib import Path
import csv
import sys
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_activity as ca
import check_dead_activity as cd
import check_dead_events as ce
import check_confirmation as cc
import possession
from starter.load_smart_tagging import load

FPS = ca.FPS


def near_set_piece(episodes, windows):
    kept = []
    for s, e in episodes:
        if any(ws - ce.RADIUS <= e and we + ce.RADIUS >= s for ws, we in windows):
            kept.append((s, e))
    return kept


def structure(name, submission, graded_total):
    lengths = cc.spell_lengths(submission)
    d_frames = sum(label == "D" for label in submission)

    switches = sum(
        1
        for a, b in zip(submission, submission[1:])
        if a in ("H", "A") and b in ("H", "A") and a != b
    )

    print(
        f"{name:<24} | {100 * d_frames / graded_total:>5.1f}% | "
        f"{len(lengths):>7} | {median(lengths) / FPS:>11.2f} | "
        f"{100 * sum(x <= FPS for x in lengths) / len(lengths):>4.0f}% | "
        f"{100 * sum(x <= 2 * FPS for x in lengths) / len(lengths):>4.0f}% | "
        f"{max(lengths) / FPS:>8.0f} | {switches:>8}"
    )


def main():
    xs, ys = ca.read_players()
    speed = ca.median_player_speed(xs, ys)
    valid = ca.read_ball_valid()
    n = len(valid)

    graded = np.zeros(n, dtype=bool)
    for a, b in ca.EVALUATED_INTERVALS:
        graded[a:b + 1] = True
    graded_total = int(graded.sum())

    pause = np.zeros(n, dtype=bool)
    pause[ca.OFFICIAL_PAUSE[0]:ca.OFFICIAL_PAUSE[1] + 1] = True
    pause_speed = speed[pause & np.isfinite(speed)]

    windows = [
        (e["start_frame"], e["end_frame"])
        for e in load()
        if str(e.get("code", "")).strip().upper() == "SET PIECES"
    ]

    with possession.FEATURES_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    dead_v5 = np.array(possession.read_dead(possession.DEAD_PATH))

    print("varianta                 | D eval | posesii | mediana (s) | <=1s | <=2s | max (s)  | schimb. H/A")
    print("-------------------------+--------+---------+-------------+------+------+----------+-----------")

    structure("V5 (actual)", cc.build_submission(rows, dead_v5.tolist()), graded_total)

    for pct in (50, 75):
        eps = ce.episodes_for(speed, valid, graded, pause_speed, 10, pct)
        kept = near_set_piece(eps, windows)

        extra = np.zeros(n, dtype=bool)
        for s, e in kept:
            extra[s:e + 1] = True

        dead_v6 = dead_v5 | (extra & graded)
        added = int((dead_v6 & ~dead_v5).sum())

        structure(
            f"V6 gol>10s, <=p{pct}",
            cc.build_submission(rows, dead_v6.tolist()),
            graded_total,
        )
        print(
            f"{'':<24}   (episoade pastrate: {len(kept)}/{len(eps)}, "
            f"cadre D adaugate: {added})"
        )


if __name__ == "__main__":
    main()