
from pathlib import Path
import csv
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "possession_ha.csv"

EVALUATED_INTERVALS = [
    (11185, 90651),
    (110808, 190197),
]

FPS = 25


def get_spells(labels, start, end):
    """Grupează cadrele consecutive cu aceeași etichetă."""
    spells = []
    spell_start = start
    current = labels[start]

    for frame in range(start + 1, end + 2):
        new_label = labels[frame] if frame <= end else None

        if new_label != current:
            spells.append({
                "start": spell_start,
                "end": frame - 1,
                "label": current,
                "frames": frame - spell_start,
            })

            if frame <= end:
                spell_start = frame
                current = new_label

    return spells


def main():
    with INPUT.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    labels = [row["label"] for row in rows]

    assert len(labels) == 199319
    assert all(int(row["frame"]) == i for i, row in enumerate(rows))

    spells = []

    for start, end in EVALUATED_INTERVALS:
        spells.extend(get_spells(labels, start, end))

    possession_spells = [
        s for s in spells if s["label"] in ("H", "A")
    ]

    durations = [s["frames"] for s in possession_spells]

    short_10 = [s for s in possession_spells if s["frames"] <= 10]
    short_25 = [s for s in possession_spells if s["frames"] <= 25]

    transitions = sum(
        1
        for start, end in EVALUATED_INTERVALS
        for frame in range(start + 1, end + 1)
        if labels[frame] in ("H", "A")
        and labels[frame - 1] in ("H", "A")
        and labels[frame] != labels[frame - 1]
    )

    print("=== DIAGNOSTIC H/A ===")
    print("Intervale H/A:", len(possession_spells))
    print("Tranzitii H/A evaluate:", transitions)
    print("Durata mediana:", round(median(durations) / FPS, 2), "sec")
    print("Durata minima:", round(min(durations) / FPS, 2), "sec")
    print("Durata maxima:", round(max(durations) / FPS, 2), "sec")
    print("Intervale <= 0.4 sec:", len(short_10))
    print("Intervale <= 1 sec:", len(short_25))

    print("\nPrimele 15 intervale scurte (<= 1 sec):")
    for spell in short_25[:15]:
        print(spell)


def inspect_short_spells():
    features_path = ROOT / "features_ball.csv"

    with features_path.open(newline="", encoding="utf-8") as file:
        features = list(csv.DictReader(file))

    with INPUT.open(newline="", encoding="utf-8") as file:
        labels = [row["label"] for row in csv.DictReader(file)]

    spells = []
    for start, end in EVALUATED_INTERVALS:
        spells.extend(get_spells(labels, start, end))

    short_spells = [
        s for s in spells
        if s["label"] in ("H", "A")
        and s["frames"] <= 25
    ]

    print("\n=== DOVEZI PENTRU PRIMELE 10 INTERVALE SCURTE ===")

    for spell in short_spells[:10]:
        start, end = spell["start"], spell["end"]
        segment = features[start:end + 1]

        valid = [
            row for row in segment
            if row["ball_valid"] == "True"
        ]

        supporting = 0

        for row in valid:
            margin = float(row["distance_margin_m"])

            if spell["label"] == "H" and margin > 0:
                supporting += 1
            elif spell["label"] == "A" and margin < 0:
                supporting += 1

        print(
            f"{spell['label']} {start}-{end}: "
            f"{spell['frames']} cadre, "
            f"minge vizibila {len(valid)}/{len(segment)}, "
            f"avantaj geometric {supporting}/{len(valid)}"
        )

def inspect_transition(start=11565, end=11590):
    with (ROOT / "features_ball.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        features = list(csv.DictReader(file))

    with INPUT.open(newline="", encoding="utf-8") as file:
        labels = [row["label"] for row in csv.DictReader(file)]

    print(f"\n=== CADRE {start}-{end} ===")
    print("Frame | Pos | Home(m) | Away(m) | Margin(m)")

    for frame in range(start, end + 1):
        row = features[frame]

        print(
            f"{frame:5d} | "
            f"{labels[frame]:3s} | "
            f"{float(row['nearest_home_m']):7.2f} | "
            f"{float(row['nearest_away_m']):7.2f} | "
            f"{float(row['distance_margin_m']):9.2f}"
        )

def inspect_ball_motion(start=11565, end=11590):
    import math

    with (ROOT / "features_ball.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        features = list(csv.DictReader(file))

    print("\n=== TRAIECTORIA MINGII ===")
    print("Frame | X      | Y      | Speed(m/s) | Delta speed")

    previous_velocity = None

    for frame in range(start, end + 1):
        row = features[frame]
        previous = features[frame - 1]

        x = float(row["ball_x"])
        y = float(row["ball_y"])

        speed = float("nan")
        delta_speed = float("nan")

        if (
            math.isfinite(x)
            and math.isfinite(y)
            and previous["ball_valid"] == "True"
        ):
            prev_x = float(previous["ball_x"])
            prev_y = float(previous["ball_y"])

            vx = (x - prev_x) * 105.3 * FPS
            vy = (y - prev_y) * 68.0 * FPS

            speed = math.hypot(vx, vy)

            if previous_velocity is not None:
                delta_speed = math.hypot(
                    vx - previous_velocity[0],
                    vy - previous_velocity[1],
                )

            previous_velocity = (vx, vy)
        else:
            previous_velocity = None

        print(
            f"{frame:5d} | "
            f"{x:6.3f} | "
            f"{y:6.3f} | "
            f"{speed:10.2f} | "
            f"{delta_speed:11.2f}"
        )

def inspect_short_spell_motion():
    import math

    with (ROOT / "features_ball.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        features = list(csv.DictReader(file))

    with INPUT.open(newline="", encoding="utf-8") as file:
        labels = [row["label"] for row in csv.DictReader(file)]

    spells = []
    for start, end in EVALUATED_INTERVALS:
        spells.extend(get_spells(labels, start, end))

    short_spells = [
        s for s in spells
        if s["label"] in ("H", "A")
        and s["frames"] <= 25
    ]

    print("\n=== MISCAREA MINGII IN INTERVALE SCURTE ===")
    print("Start | Team | Durata | Max delta-v (m/s)")

    for spell in short_spells[:15]:
        start = spell["start"]
        end = spell["end"]

        max_delta_v = 0.0
        previous_velocity = None

        for frame in range(max(1, start - 5), min(len(features), end + 6)):
            row = features[frame]
            prev = features[frame - 1]

            if (
                row["ball_valid"] != "True"
                or prev["ball_valid"] != "True"
            ):
                previous_velocity = None
                continue

            vx = (
                float(row["ball_x"])
                - float(prev["ball_x"])
            ) * 105.3 * FPS

            vy = (
                float(row["ball_y"])
                - float(prev["ball_y"])
            ) * 68.0 * FPS

            if previous_velocity is not None:
                delta_v = math.hypot(
                    vx - previous_velocity[0],
                    vy - previous_velocity[1],
                )
                max_delta_v = max(max_delta_v, delta_v)

            previous_velocity = (vx, vy)

        print(
            f"{start:5d} | "
            f"{spell['label']:4s} | "
            f"{spell['frames']:6d} | "
            f"{max_delta_v:17.2f}"
        )

def compare_motion_by_duration():
    import math
    from statistics import median

    with (ROOT / "features_ball.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        features = list(csv.DictReader(file))

    with INPUT.open(newline="", encoding="utf-8") as file:
        labels = [row["label"] for row in csv.DictReader(file)]

    # Precalculăm variația vectorului de viteză.
    delta_v = [float("nan")] * len(features)
    velocities = [None] * len(features)

    for frame in range(1, len(features)):
        current = features[frame]
        previous = features[frame - 1]

        if (
            current["ball_valid"] != "True"
            or previous["ball_valid"] != "True"
        ):
            continue

        vx = (
            float(current["ball_x"]) -
            float(previous["ball_x"])
        ) * 105.3 * FPS

        vy = (
            float(current["ball_y"]) -
            float(previous["ball_y"])
        ) * 68.0 * FPS

        velocities[frame] = (vx, vy)

        if velocities[frame - 1] is not None:
            prev_vx, prev_vy = velocities[frame - 1]
            delta_v[frame] = math.hypot(
                vx - prev_vx,
                vy - prev_vy,
            )

    spells = []
    for start, end in EVALUATED_INTERVALS:
        spells.extend(get_spells(labels, start, end))

    short = []
    long = []

    for spell in spells:
        if spell["label"] not in ("H", "A"):
            continue

        start = max(0, spell["start"] - 5)
        end = min(len(delta_v), spell["end"] + 6)

        values = [
            v for v in delta_v[start:end]
            if math.isfinite(v)
        ]

        if not values:
            continue

        maximum = max(values)

        if spell["frames"] <= 25:
            short.append(maximum)
        elif spell["frames"] >= 125:
            long.append(maximum)

    print("\n=== COMPARATIE MISCARE ===")

    for name, values in [
        ("Scurte (<= 1 sec)", short),
        ("Lungi (>= 5 sec)", long),
    ]:
        if values:
            print(
                f"{name}: "
                f"n={len(values)}, "
                f"mediana max delta-v={median(values):.2f} m/s"
            )




if __name__ == "__main__":
    main()
    # inspect_short_spells()
    # inspect_transition()
    # inspect_ball_motion()
    # inspect_short_spell_motion()
    compare_motion_by_duration()