
from pathlib import Path
import csv
import math
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
FPS = 25
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
RADIUS = 5

INTERVALS = [(11185, 90651), (110808, 190197)]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    features = read_csv(ROOT / "features_ball.csv")
    possession = read_csv(ROOT / "possession_ha.csv")
    labels = [row["label"] for row in possession]

    assert len(features) == len(labels) == 199319

    # Viteza în planul terenului.
    velocities = [None] * len(features)

    for i in range(1, len(features)):
        if not (
            features[i]["ball_valid"] == "True"
            and features[i - 1]["ball_valid"] == "True"
        ):
            continue

        dx = (
            float(features[i]["ball_x"])
            - float(features[i - 1]["ball_x"])
        )

        dy = (
            float(features[i]["ball_y"])
            - float(features[i - 1]["ball_y"])
        )

        velocities[i] = (
            dx * PITCH_LENGTH * FPS,
            dy * PITCH_WIDTH * FPS,
        )

    delta_v = [float("nan")] * len(features)

    for i in range(1, len(features)):
        if velocities[i] is None or velocities[i - 1] is None:
            continue

        delta_v[i] = math.dist(
            velocities[i],
            velocities[i - 1],
        )

    transitions = []

    for start, end in INTERVALS:
        for i in range(start + RADIUS, end - RADIUS + 1):
            if (
                labels[i] not in ("H", "A")
                or labels[i - 1] not in ("H", "A")
                or labels[i] == labels[i - 1]
            ):
                continue

            window = delta_v[i - RADIUS:i + RADIUS + 1]

            if not all(math.isfinite(v) for v in window):
                continue

            transitions.append({
                "frame": i,
                "from": labels[i - 1],
                "to": labels[i],
                "max_delta_v": max(window),
            })

    values = [t["max_delta_v"] for t in transitions]

    print("=== TRANZITII H/A - FERESTRE EGALE ===")
    print("Tranzitii cu date complete:", len(values))

    if not values:
        return

    print("Mediana max delta-v:", round(median(values), 3))
    print("Minim:", round(min(values), 3))
    print("Maxim:", round(max(values), 3))

    print("\nPrimele 15 tranzitii:")
    for transition in transitions[:15]:
        print(transition)


if __name__ == "__main__":
    main()
