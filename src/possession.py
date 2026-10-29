
from pathlib import Path
import csv
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "features_ball.csv"
OUTPUT_PATH = PROJECT_ROOT / "possession_ha.csv"

EXPECTED_FRAMES = 199_319

# Parametri inițiali, euristici — NU optimizați pe ground truth.
MAX_CONTROL_DISTANCE_M = 2.0
MIN_DISTANCE_MARGIN_M = 0.75
CONFIRMATION_FRAMES = 5


DEAD_PATH = PROJECT_ROOT / "dead_ball_candidates.csv"


def read_dead(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} lipseste. Ruleaza intai: python src/dead_ball.py"
        )

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != EXPECTED_FRAMES:
        raise ValueError(f"Numar incorect de cadre D: {len(rows)}")

    return [row["dead_candidate"] == "1" for row in rows]

def candidate_team(row):
    """
    Identifică echipa cu dovezi geometrice clare.

    Returnează:
        H - posibil control Barcelona
        A - posibil control Athletic
        None - observație insuficientă
    """
    if row["ball_valid"] != "True":
        return None

    home = float(row["nearest_home_m"])
    away = float(row["nearest_away_m"])

    if not (math.isfinite(home) and math.isfinite(away)):
        return None

    margin = away - home

    if (
        home <= MAX_CONTROL_DISTANCE_M
        and margin >= MIN_DISTANCE_MARGIN_M
    ):
        return "H"

    if (
        away <= MAX_CONTROL_DISTANCE_M
        and margin <= -MIN_DISTANCE_MARGIN_M
    ):
        return "A"

    return None


def estimate_possession(rows, dead):
    """
    Automat de stări H/A.

    U = posesie încă necunoscută (doar rezultat intermediar). la fiecare cadru D automat este resetat: dupa oprire, posesia este reluata doar din dovezi noi
    """
    labels = []
    current_team = None

    pending_team = None
    pending_start = None
    pending_count = 0

    transitions = []

    for index, row in enumerate(rows):
        if dead[index]:
            current_team = None
            pending_team = None
            pending_start = None
            pending_count = 0
            labels.append("U")
            continue
        candidate = candidate_team(row)

        # Încă nu avem o posesie inițială.
        if current_team is None:
            if candidate is None:
                labels.append("U")
                continue

            current_team = candidate
            labels.append(current_team)
            continue

        # Nicio dovadă nouă sau aceeași echipă.
        if candidate is None or candidate == current_team:
            pending_team = None
            pending_start = None
            pending_count = 0

            labels.append(current_team)
            continue

        # Apare un posibil control al adversarului.
        if candidate != pending_team:
            pending_team = candidate
            pending_start = index
            pending_count = 1
        else:
            pending_count += 1

        # Până la confirmare păstrăm posesia anterioară.
        labels.append(current_team)

        if pending_count >= CONFIRMATION_FRAMES:
            previous_team = current_team
            current_team = pending_team

            # Confirmarea vine acum, dar atribuim tranziția
            # primului cadru al secvenței candidate.
            for j in range(pending_start, index + 1):
                labels[j] = current_team

            transitions.append({
                "frame": pending_start,
                "from": previous_team,
                "to": current_team,
            })

            pending_team = None
            pending_start = None
            pending_count = 0

    return labels, transitions


def main():
    with FEATURES_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    if len(rows) != EXPECTED_FRAMES:
        raise ValueError(
            f"Așteptate {EXPECTED_FRAMES} cadre, "
            f"găsite {len(rows)}"
        )

    for index, row in enumerate(rows):
        if int(row["frame"]) != index:
            raise ValueError(
                f"Index incorect la cadrul {index}"
            )
    
    
    dead = read_dead(DEAD_PATH)
    labels, transitions = estimate_possession(rows, dead)

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "label"])

        for frame, label in enumerate(labels):
            writer.writerow([frame, label])

    print("Cadre procesate:", len(labels))
    print("Cadre H:", labels.count("H"))
    print("Cadre A:", labels.count("A"))
    print("Cadre U:", labels.count("U"))
    print("Tranziții H/A confirmate:", len(transitions))
    print("Fișier generat:", OUTPUT_PATH)

    print("\nPrimele 10 tranziții:")
    for transition in transitions[:10]:
        print(transition)

    assert len(labels) == EXPECTED_FRAMES
    assert set(labels).issubset({"H", "A", "U"})

    print("\nVerificările au trecut!")


if __name__ == "__main__":
    main()
