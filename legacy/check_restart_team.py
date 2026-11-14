from pathlib import Path
import csv
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starter.load_smart_tagging import load

FPS = 25

# Ferestre exploratorii, nu praguri validate.
LOOKBACK = 10 * FPS
LOOKAHEAD = 30 * FPS

TEAM_TO_LABEL = {"barcelona": "H", "atletic club": "A"}
TYPES = ("Corner Kick", "Goal Kick", "Free Kick", "Kick Off")


def read_column(path, column):
    with path.open(newline="", encoding="utf-8") as file:
        return [row[column] for row in csv.DictReader(file)]


def main():
    labels = read_column(ROOT / "possession_ha.csv", "label")
    dead = [
        value == "1"
        for value in read_column(
            ROOT / "dead_ball_candidates.csv", "dead_candidate"
        )
    ]
    n_frames = len(labels)

    stats = defaultdict(lambda: defaultdict(int))

    for event in load():
        if str(event.get("code", "")).strip().upper() != "SET PIECES":
            continue

        tags = event.get("tags", {})
        event_type = tags.get("Type", "")

        if not any(t in event_type for t in TYPES):
            continue

        expected = TEAM_TO_LABEL.get(
            str(tags.get("Team", "")).strip().lower()
        )

        start = max(0, event["start_frame"] - LOOKBACK)
        end = min(n_frames, event["end_frame"])

        first_dead = next(
            (f for f in range(start, end) if dead[f]), None
        )

        if first_dead is None:
            stats[event_type]["fara D in fereastra"] += 1
            continue

        # Sfarsitul primului episod D consecutiv.
        last_dead = first_dead
        while last_dead + 1 < n_frames and dead[last_dead + 1]:
            last_dead += 1

        # Prima echipa H/A dupa oprire (automatul e resetat la D).
        first_team = next(
            (
                labels[f]
                for f in range(
                    last_dead + 1,
                    min(n_frames, last_dead + 1 + LOOKAHEAD),
                )
                if labels[f] in ("H", "A")
            ),
            None,
        )

        if expected is None:
            stats[event_type]["tag Team = N/A"] += 1
        elif first_team is None:
            stats[event_type]["fara dovezi dupa D"] += 1
        elif first_team == expected:
            stats[event_type]["acord"] += 1
        else:
            stats[event_type]["dezacord"] += 1

    print("=== TAG Team vs PRIMA ECHIPA DUPA D ===")
    for event_type in sorted(stats):
        print(f"\n{event_type}")
        for key, value in sorted(stats[event_type].items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()