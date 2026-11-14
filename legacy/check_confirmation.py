from pathlib import Path
import csv
from statistics import median

import possession
import predict

ROOT = Path(__file__).resolve().parents[1]
FPS = 25

# 0.2 s (valoarea actuala) pana la 5 s.
CANDIDATES = [5, 12, 25, 50, 75, 125]


def build_submission(rows, dead):
    labels, _ = possession.estimate_possession(rows, dead)

    submission = ["X"] * predict.N_FRAMES

    for start, end in predict.EVALUATED_INTERVALS:
        submission[start:end + 1] = predict.fill_unknown(labels, start, end)

    for frame, is_dead in enumerate(dead):
        if is_dead and submission[frame] != "X":
            submission[frame] = "D"

    return submission


def spell_lengths(submission):
    lengths = []
    start = 0

    for frame in range(1, len(submission) + 1):
        if frame == len(submission) or submission[frame] != submission[start]:
            if submission[start] in ("H", "A"):
                lengths.append(frame - start)
            start = frame

    return lengths


def main():
    with possession.FEATURES_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    dead = possession.read_dead(possession.DEAD_PATH)

    print("confirm | s    | posesii | mediana (s) | <=1 s | <=2 s")
    print("--------+------+---------+-------------+-------+------")

    for n in CANDIDATES:
        possession.CONFIRMATION_FRAMES = n

        lengths = spell_lengths(build_submission(rows, dead))
        total = len(lengths)

        print(
            f"{n:>7} | {n / FPS:>4.1f} | {total:>7} | "
            f"{median(lengths) / FPS:>11.2f} | "
            f"{100 * sum(x <= FPS for x in lengths) / total:>4.0f}% | "
            f"{100 * sum(x <= 2 * FPS for x in lengths) / total:>4.0f}%"
        )


if __name__ == "__main__":
    main()
    