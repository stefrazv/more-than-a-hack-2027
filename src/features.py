
from pathlib import Path
import csv
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from starter.baseline import team_of_each_track

ATD_PATH = (
    PROJECT_ROOT / "data" / "tactical"
    / "BAR-ATH_tactical_atd.txt"
)

META_PATH = (
    PROJECT_ROOT / "data" / "tactical"
    / "BAR-ATH_tactical_atd.xml"
)

OUTPUT_PATH = PROJECT_ROOT / "features_ball.csv"

EXPECTED_FRAMES = 199_319
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0


def nearest_team_distances(
    players_text, ball_x, ball_y, team_of_track
):
    """Distanța minimă în metri până la fiecare echipă."""

    nearest_home = math.inf
    nearest_away = math.inf

    player_groups = players_text.split(";")

    if len(player_groups) != len(team_of_track):
        raise ValueError(
            "Numărul sloturilor nu corespunde metadatelor"
        )

    for slot, group in enumerate(player_groups):
        fields = group.split(",")

        if len(fields) != 3:
            raise ValueError(f"Slot invalid: {slot}")

        player_x = float(fields[0])
        player_y = float(fields[1])

        if not (
            math.isfinite(player_x)
            and math.isfinite(player_y)
        ):
            continue

        dx = (player_x - ball_x) * PITCH_LENGTH
        dy = (player_y - ball_y) * PITCH_WIDTH

        distance = math.hypot(dx, dy)

        if team_of_track[slot] == 0:
            nearest_home = min(nearest_home, distance)
        else:
            nearest_away = min(nearest_away, distance)

    return (
        nearest_home if math.isfinite(nearest_home) else math.nan,
        nearest_away if math.isfinite(nearest_away) else math.nan,
    )


def read_features(atd_path, meta_path):
    """Extrage caracteristicile geometrice pe cadru."""

    team_of_track = team_of_each_track(meta_path)

    with atd_path.open("r", encoding="utf-8") as file:
        for expected_frame, line in enumerate(file):
            parts = line.strip().split(":")

            if len(parts) != 3:
                raise ValueError(
                    f"Format ATD invalid la linia {expected_frame + 1}"
                )

            frame_id = int(parts[0])
            video_frame = frame_id - 1

            if video_frame != expected_frame:
                raise ValueError(
                    f"Cadru neașteptat: {video_frame}"
                )

            ball_fields = parts[2].split(",")

            if len(ball_fields) != 3:
                raise ValueError(
                    f"Format minge invalid la cadrul {video_frame}"
                )

            ball_x = float(ball_fields[0])
            ball_y = float(ball_fields[1])

            ball_valid = (
                math.isfinite(ball_x)
                and math.isfinite(ball_y)
            )

            ball_outside = (
                ball_valid
                and (
                    ball_x < 0 or ball_x > 1
                    or ball_y < 0 or ball_y > 1
                )
            )

            nearest_home = math.nan
            nearest_away = math.nan
            distance_margin = math.nan

            if ball_valid:
                nearest_home, nearest_away = (
                    nearest_team_distances(
                        parts[1],
                        ball_x,
                        ball_y,
                        team_of_track,
                    )
                )

                if (
                    math.isfinite(nearest_home)
                    and math.isfinite(nearest_away)
                ):
                    distance_margin = (
                        nearest_away - nearest_home
                    )

            yield {
                "frame": video_frame,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_valid": ball_valid,
                "ball_outside": ball_outside,
                "nearest_home_m": nearest_home,
                "nearest_away_m": nearest_away,
                "distance_margin_m": distance_margin,
            }


def main():
    columns = [
        "frame",
        "ball_x",
        "ball_y",
        "ball_valid",
        "ball_outside",
        "nearest_home_m",
        "nearest_away_m",
        "distance_margin_m",
    ]

    frame_count = 0
    valid_count = 0
    outside_count = 0
    distance_count = 0

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=columns
        )
        writer.writeheader()

        for row in read_features(ATD_PATH, META_PATH):
            writer.writerow(row)

            frame_count += 1
            valid_count += int(row["ball_valid"])
            outside_count += int(row["ball_outside"])

            if math.isfinite(row["distance_margin_m"]):
                distance_count += 1

    print("Cadre procesate:", frame_count)
    print("Minge detectată:", valid_count)
    print("Minge în exterior:", outside_count)
    print("Cadre cu distanțe pentru ambele echipe:", distance_count)
    print("Fișier generat:", OUTPUT_PATH)

    assert frame_count == EXPECTED_FRAMES
    assert valid_count == 113_488
    assert outside_count == 4_964

    print("Toate verificările au trecut!")


if __name__ == "__main__":
    main()
