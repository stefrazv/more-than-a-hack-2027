
from pathlib import Path
import sys
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starter.load_smart_tagging import load

FEATURES_PATH = ROOT / "features_ball.csv"
FPS = 25

# Fereastră exploratorie, nu un prag validat.
SEARCH_RADIUS = 15 * FPS

EVALUATED_INTERVALS = [
    (11185, 90651),
    (110808, 190197),
]


def is_evaluated(frame):
    return any(
        start <= frame <= end
        for start, end in EVALUATED_INTERVALS
    )


def load_outside_episodes():
    """Grupează cadrele consecutive cu mingea în exterior."""
    with FEATURES_PATH.open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    episodes = []
    start = None

    for frame, row in enumerate(rows):
        outside = (
            row["ball_outside"] == "True"
            and is_evaluated(frame)
        )

        if outside and start is None:
            start = frame

        if not outside and start is not None:
            episodes.append((start, frame - 1))
            start = None

    if start is not None:
        episodes.append((start, len(rows) - 1))

    return episodes


def load_restarts():
    """Selectează evenimentele SET PIECES."""
    events = load()

    restarts = [
        event
        for event in events
        if (event["code"] or "").strip().upper()
        == "SET PIECES"
    ]

    return sorted(
        restarts,
        key=lambda event: event["start_frame"]
    )


def episode_gap(event, episode):
    """
    Distanța dintre două intervale, în cadre.

    0 înseamnă suprapunere.
    """
    event_start = event["start_frame"]
    event_end = event["end_frame"] - 1

    outside_start, outside_end = episode

    if outside_end < event_start:
        return event_start - outside_end

    if event_end < outside_start:
        return outside_start - event_end

    return 0


def main():
    restarts = load_restarts()
    episodes = load_outside_episodes()

    print("=== DATE DISPONIBILE PENTRU D ===")
    print("SET PIECES:", len(restarts))
    print("Episoade ball_outside:", len(episodes))

    matched = 0

    print("\n=== PRIMELE 20 DE RELUARI ===")
    print(
        "Tip | Start | End | "
        "Outside start-end | Distanta (s)"
    )

    for event in restarts:
        if not episodes:
            break

        nearest = min(
            episodes,
            key=lambda ep: episode_gap(event, ep)
        )

        gap = episode_gap(event, nearest)

        if gap <= SEARCH_RADIUS:
            matched += 1

        if restarts.index(event) >= 20:
            continue

        event_type = event["tags"].get("Type", "?")

        print(
            f"{event_type} | "
            f"{event['start_frame']} | "
            f"{event['end_frame']} | "
            f"{nearest[0]}-{nearest[1]} | "
            f"{gap / FPS:.2f}"
        )

    print("\nReluari cu minge in exterior")
    print(f"la cel mult 15 secunde: {matched}/{len(restarts)}")


def generate_dead_ball_candidates():
    output_path = ROOT / "dead_ball_candidates.csv"

    with FEATURES_PATH.open(
        newline="", encoding="utf-8"
    ) as file:
        features = list(csv.DictReader(file))

    restarts = load_restarts()
    n_frames = len(features)

    # Marcăm cadrele acoperite de cel puțin un SET PIECES.
    set_piece_mask = [False] * n_frames

    for event in restarts:
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])

        # Intervalele Smart Tagging sunt [start, end).
        for frame in range(start, end):
            set_piece_mask[frame] = True

    # Intersecția dintre cele două indicii.
    candidates = [False] * n_frames

    for frame, row in enumerate(features):
        candidates[frame] = (
            is_evaluated(frame)
            and row["ball_outside"] == "True"
            and set_piece_mask[frame]
        )

    # Numărăm episoadele consecutive candidate.
    episodes = []
    start = None

    for frame, is_candidate in enumerate(candidates):
        if is_candidate and start is None:
            start = frame

        if start is not None and (
            not is_candidate or frame == n_frames - 1
        ):
            end = frame - 1 if not is_candidate else frame
            episodes.append((start, end))
            start = None

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "dead_candidate"])

        for frame, candidate in enumerate(candidates):
            writer.writerow([frame, int(candidate)])

    print("\n=== CANDIDATI D ===")
    print("Cadre candidate:", sum(candidates))
    print("Episoade candidate:", len(episodes))
    print("Fisier generat:", output_path)

    print("\nPrimele 10 episoade:")
    for start, end in episodes[:10]:
        print(
            f"{start}-{end}: "
            f"{(end - start + 1) / FPS:.2f} sec"
        )

if __name__ == "__main__":
    main()
    generate_dead_ball_candidates()
