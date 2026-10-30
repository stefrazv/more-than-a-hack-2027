from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "features_ball.csv"

FPS = 25
EVALUATED_INTERVALS = [(11185, 90651), (110808, 190197)]

BINS = [
    (0.5, "<= 0.5 s"),
    (1, "0.5 - 1 s"),
    (2, "1 - 2 s"),
    (5, "2 - 5 s"),
    (10, "5 - 10 s"),
    (float("inf"), "> 10 s"),
]


def main():
    with FEATURES.open(newline="", encoding="utf-8") as file:
        valid = [row["ball_valid"] == "True" for row in csv.DictReader(file)]

    gaps = []

    for start, end in EVALUATED_INTERVALS:
        run = 0

        for frame in range(start, end + 1):
            if valid[frame]:
                if run:
                    gaps.append(run)
                    run = 0
            else:
                run += 1

        if run:
            gaps.append(run)

    total = sum(gaps)

    print("Goluri de minge in intervalele evaluate:", len(gaps))
    print("Cadre fara minge:", total)
    print("Cel mai lung gol (s):", round(max(gaps) / FPS, 1))
    print()
    print("durata gol  | goluri | cadre | % din cadrele lipsa")
    print("------------+--------+-------+--------------------")

    lower = 0.0

    for limit, name in BINS:
        selected = [g for g in gaps if lower < g / FPS <= limit]
        frames = sum(selected)

        print(
            f"{name:<11} | {len(selected):>6} | {frames:>5} | "
            f"{100 * frames / total:>5.1f}%"
        )

        lower = limit


if __name__ == "__main__":
    main()