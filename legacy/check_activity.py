from pathlib import Path
import csv
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import asset_paths

FEATURES = ROOT / "features_ball.csv"

FPS = 25
PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0
EVALUATED_INTERVALS = [(11185, 90651), (110808, 190197)]
PLAY_INTERVALS = [(11185, 90651), (113742, 190197)]
OFFICIAL_PAUSE = (110808, 113741)  # D oficial, din FAQ
MID_GAP_S = 5
LONG_GAP_S = 10


def read_players():
    xs, ys = [], []

    with open(asset_paths()["atd_raw"], encoding="utf-8") as file:
        for line in file:
            groups = line.strip().split(":")[1].split(";")
            row_x, row_y = [], []

            for group in groups:
                x, y = group.split(",")[:2]
                row_x.append(float(x))
                row_y.append(float(y))

            xs.append(row_x)
            ys.append(row_y)

    return (
        np.array(xs, dtype=np.float32) * PITCH_LENGTH_M,
        np.array(ys, dtype=np.float32) * PITCH_WIDTH_M,
    )


def median_player_speed(xs, ys):
    """Mediana vitezei jucatorilor (m/s), pe fereastra de 1 s.

    Mediana peste sloturi ramane robusta la schimbari de identitate.
    """
    speed = np.full(len(xs), np.nan)
    dist = np.hypot(xs[FPS:] - xs[:-FPS], ys[FPS:] - ys[:-FPS])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed[FPS:] = np.nanmedian(dist, axis=1)

    return speed


def read_ball_valid():
    with FEATURES.open(newline="", encoding="utf-8") as file:
        return np.array(
            [row["ball_valid"] == "True" for row in csv.DictReader(file)]
        )


def describe(name, speed, mask, reference):
    values = speed[mask & np.isfinite(speed)]

    if not len(values):
        print(f"{name:<22} | fara date")
        return

    print(
        f"{name:<22} | {len(values):>6} | "
        f"{np.median(values):>5.2f} | "
        f"{np.percentile(values, 25):>4.2f}-{np.percentile(values, 75):>4.2f} | "
        f"{100 * np.mean(values <= reference):>5.0f}%"
    )


def main():
    xs, ys = read_players()
    speed = median_player_speed(xs, ys)
    valid = read_ball_valid()
    n = len(valid)

    graded = np.zeros(n, dtype=bool)
    visible = np.zeros(n, dtype=bool)
    gap_mid = np.zeros(n, dtype=bool)
    gap_long = np.zeros(n, dtype=bool)

    for start, end in PLAY_INTERVALS:
        graded[start:end + 1] = True
        run_start = None

        for frame in range(start, end + 2):
            missing = frame <= end and not valid[frame]

            if missing and run_start is None:
                run_start = frame

            if not missing and run_start is not None:
                seconds = (frame - run_start) / FPS

                if seconds > LONG_GAP_S:
                    gap_long[run_start:frame] = True
                elif seconds > MID_GAP_S:
                    gap_mid[run_start:frame] = True

                run_start = None

    visible = graded & valid

    pause = np.zeros(n, dtype=bool)
    pause[OFFICIAL_PAUSE[0]:OFFICIAL_PAUSE[1] + 1] = True

    pause_values = speed[pause & np.isfinite(speed)]
    reference = np.percentile(pause_values, 75)

    print("Viteza mediana a jucatorilor (m/s), fereastra 1 s")
    print(f"Referinta: p75 in pauza oficiala D = {reference:.2f} m/s\n")
    print("grup                   |  cadre | mediana | p25-p75   | <= ref")
    print("-----------------------+--------+---------+-----------+-------")

    describe("pauza oficiala (D)", speed, pause, reference)
    describe("minge vizibila", speed, visible, reference)
    describe("gol 5-10 s", speed, gap_mid, reference)
    describe("gol > 10 s", speed, gap_long, reference)


if __name__ == "__main__":
    main()