from pathlib import Path
import csv
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission_v4.csv"

FPS = 25
HA = ("H", "A")


def read_labels(path):
    with path.open(newline="", encoding="utf-8") as file:
        return [row["label"] for row in csv.DictReader(file)]


def get_runs(labels):
    """Segmente maximale consecutive: (eticheta, start, end)."""
    runs = []
    start = 0

    for frame in range(1, len(labels) + 1):
        if frame == len(labels) or labels[frame] != labels[start]:
            runs.append((labels[start], start, frame - 1))
            start = frame

    return runs


def share(count, total):
    return f"{count}/{total} ({100 * count / total:.0f}%)" if total else "0/0"


def main():
    runs = get_runs(read_labels(SUBMISSION))

    all_spells = [e - s + 1 for lab, s, e in runs if lab in HA]

    d_durations = []
    same_team = 0
    diff_team = 0
    no_neighbor = 0
    first_spell_after = []

    for i, (label, start, end) in enumerate(runs):
        if label != "D":
            continue

        d_durations.append(end - start + 1)

        before = runs[i - 1] if i > 0 else None
        after = runs[i + 1] if i + 1 < len(runs) else None

        if before and after and before[0] in HA and after[0] in HA:
            if before[0] == after[0]:
                same_team += 1
            else:
                diff_team += 1
            first_spell_after.append(after[2] - after[1] + 1)
        else:
            no_neighbor += 1

    print("=== EPISOADE D ===")
    print("Total:", len(d_durations))
    print("Mediana durata (s):", round(median(d_durations) / FPS, 2))
    print("<= 0.2 s:", share(sum(d <= 5 for d in d_durations), len(d_durations)))
    print("<= 1 s:", share(sum(d <= FPS for d in d_durations), len(d_durations)))

    compared = same_team + diff_team
    print("\n=== ECHIPA INAINTE vs DUPA D ===")
    print("Comparabile (vecini H/A):", compared)
    print("Aceeasi echipa:", share(same_team, compared))
    print("Echipe diferite:", share(diff_team, compared))
    print("Fara vecin H/A (X, limita):", no_neighbor)

    print("\n=== PRIMA POSESIE DUPA D ===")
    print("Mediana (s):", round(median(first_spell_after) / FPS, 2))
    print("<= 0.4 s:", share(sum(d <= 10 for d in first_spell_after), len(first_spell_after)))
    print("<= 1 s:", share(sum(d <= FPS for d in first_spell_after), len(first_spell_after)))

    print("\n=== TOATE POSESIILE H/A (referinta) ===")
    print("Numar:", len(all_spells))
    print("Mediana (s):", round(median(all_spells) / FPS, 2))
    print("<= 0.4 s:", share(sum(d <= 10 for d in all_spells), len(all_spells)))
    print("<= 1 s:", share(sum(d <= FPS for d in all_spells), len(all_spells)))


if __name__ == "__main__":
    main()