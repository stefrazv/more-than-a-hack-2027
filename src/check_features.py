
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from starter.baseline import (
    nearest_player_labels,
    team_of_each_track,
)

ATD_PATH = (
    PROJECT_ROOT / "data" / "tactical"
    / "BAR-ATH_tactical_atd.txt"
)

META_PATH = (
    PROJECT_ROOT / "data" / "tactical"
    / "BAR-ATH_tactical_atd.xml"
)

FEATURES_PATH = PROJECT_ROOT / "features_ball.csv"

PITCH = (105.3, 68.0)
N_FRAMES = 199_319


def main():
    features = pd.read_csv(FEATURES_PATH)

    assert len(features) == N_FRAMES
    assert np.array_equal(
        features["frame"].to_numpy(),
        np.arange(N_FRAMES),
    )

    baseline = nearest_player_labels(
        ATD_PATH,
        team_of_each_track(META_PATH),
        N_FRAMES,
        PITCH,
    )

    margin = features["distance_margin_m"].to_numpy()

    valid = np.isfinite(margin)
    ties = valid & (margin == 0)
    comparable = valid & ~ties

    # Codificarea oficială: HOME=0, AWAY=1.
    predicted = np.where(margin > 0, 0, 1)

    differences = comparable & (predicted != baseline)

    print("Cadre comparabile:", int(comparable.sum()))
    print("Egalități exacte:", int(ties.sum()))
    print("Diferențe față de baseline:", int(differences.sum()))

    if differences.any():
        examples = np.flatnonzero(differences)[:10]
        print("Exemple cadre diferite:", examples.tolist())
    else:
        print("Geometria este compatibilă cu baseline-ul!")


if __name__ == "__main__":
    main()
