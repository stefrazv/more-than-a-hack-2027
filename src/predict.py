
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "possession_ha.csv"
DEAD_CANDIDATES = ROOT / "dead_ball_candidates.csv"
OUTPUT = ROOT / "submission_v3.csv"

N_FRAMES = 199_319

EVALUATED_INTERVALS = [
    (11185, 90651),
    (110808, 190197),
]

SECOND_HALF_EVALUATION_START = 110808
SECOND_HALF_KICKOFF = 113742


def read_possession():
    with INPUT.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != N_FRAMES:
        raise ValueError(
            f"Numar incorect de cadre H/A: {len(rows)}"
        )

    labels = []

    for frame, row in enumerate(rows):
        if int(row["frame"]) != frame:
            raise ValueError(
                f"Index H/A incorect la cadrul {frame}"
            )

        label = row["label"]

        if label not in {"H", "A", "U"}:
            raise ValueError(
                f"Eticheta H/A invalida la cadrul {frame}: {label}"
            )

        labels.append(label)

    return labels


def read_dead_candidates():
    with DEAD_CANDIDATES.open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    if len(rows) != N_FRAMES:
        raise ValueError(
            f"Numar incorect de cadre D: {len(rows)}"
        )

    candidates = []

    for frame, row in enumerate(rows):
        if int(row["frame"]) != frame:
            raise ValueError(
                f"Index D incorect la cadrul {frame}"
            )

        value = row["dead_candidate"]

        if value not in {"0", "1"}:
            raise ValueError(
                f"Valoare D invalida la cadrul {frame}: {value}"
            )

        candidates.append(value == "1")

    return candidates


def fill_unknown(labels, start, end):
    """
    Completeaza U folosind posesia cunoscuta
    din acelasi interval evaluat.
    """
    result = labels[start:end + 1].copy()

    first_known = next(
        (label for label in result if label in {"H", "A"}),
        None,
    )

    if first_known is None:
        raise ValueError(
            f"Nicio posesie cunoscuta in intervalul {start}-{end}"
        )

    last_known = first_known

    for index, label in enumerate(result):
        if label in {"H", "A"}:
            last_known = label
        else:
            result[index] = last_known

    return result


def main():
    possession = read_possession()
    dead_candidates = read_dead_candidates()

    # Initial, toate cadrele sunt in afara evaluarii.
    submission = ["X"] * N_FRAMES

    # 1. Completam intervalele evaluate cu H/A.
    for start, end in EVALUATED_INTERVALS:
        submission[start:end + 1] = fill_unknown(
            possession, start, end
        )

    # 2. Inainte de kickoff-ul reprizei a doua,
    # jocul este oprit.
    for frame in range(
        SECOND_HALF_EVALUATION_START,
        SECOND_HALF_KICKOFF,
    ):
        submission[frame] = "D"

    # 3. Aplicam candidatii D din dead_ball.py.
    for frame, is_dead_candidate in enumerate(dead_candidates):
        if is_dead_candidate and submission[frame] != "X":
            submission[frame] = "D"

    # 4. Verificam structura rezultatului.
    if len(submission) != N_FRAMES:
        raise ValueError("Numar incorect de cadre")

    if not set(submission).issubset({"H", "A", "D", "X"}):
        raise ValueError("Exista etichete invalide")

    evaluated_count = sum(
        end - start + 1
        for start, end in EVALUATED_INTERVALS
    )

    if sum(label != "X" for label in submission) != evaluated_count:
        raise ValueError(
            "Numarul cadrelor evaluate nu corespunde"
        )

    # 5. Scriem CSV-ul dupa aplicarea tuturor regulilor.
    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "label"])

        for frame, label in enumerate(submission):
            writer.writerow([frame, label])

    # 6. Afisam statisticile finale.
    print("=== SUBMISSION V3 ===")

    for label in ("H", "A", "D", "X"):
        print(f"Cadre {label}: {submission.count(label)}")

    print("Total cadre:", len(submission))
    print("Cadre evaluate:", evaluated_count)
    print("Fisier:", OUTPUT)
    print("Verificarile au trecut!")


if __name__ == "__main__":
    main()
