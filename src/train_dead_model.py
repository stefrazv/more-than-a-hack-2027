"""Train the dead-ball (D) model on external data and save it to trained_models/dead_ball_model.joblib.

Training data: the 7 public DFL/IDSSE Bundesliga matches (Bassek et al., 2025, CC BY 4.0),
preprocessed with PathCRF's datatools/preprocess.py. Nothing from the competition match is used.

Each match is degraded to look like the competition's automatic tracking (degrade.py:
anonymous slots, gaps, duplicates, noise) and stitched back to 11 players per team
(stitch.py), so the model sees the same kind of input it will get on the competition match.
Labels: DFL ball status (dead / alive) per frame. Model: scikit-learn
HistGradientBoostingClassifier on player-only features (dead_dfl.frame_features).

The saved model is committed to the repository, so reproducing the submission does not
require the DFL data. This script documents exactly how the model was obtained.

    python src/train_dead_model.py --dfl <folder with the 7 preprocessed *.parquet files>
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
PATHCRF = ROOT / "third_party" / "pathcrf"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PATHCRF))

from src.dead_dfl import frame_features  # noqa: E402
from src.eval_dfl import ground_truth  # noqa: E402
from src.robust_dfl import degraded_table, load, train  # noqa: E402

MODEL_PATH = ROOT / "trained_models" / "dead_ball_model.joblib"
DEGRADE_SEED = 47


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl", required=True, help="folder with the preprocessed DFL matches (*.parquet)")
    opts = parser.parse_args()
    import joblib
    from datatools.utils import calculate_running_features

    folder = Path(opts.dfl)
    matches = sorted(p.stem for p in folder.glob("*.parquet"))
    if len(matches) != 7:
        raise SystemExit(f"expected 7 DFL matches in {folder}, found {len(matches)}")

    feats, labels = [], []
    for match in matches:
        print(f"degrading and stitching {match} ...")
        df = load(folder, match)
        feats.append(frame_features(degraded_table(df, DEGRADE_SEED, calculate_running_features), False))
        labels.append(ground_truth(df)[0] == "D")

    model = train(feats, labels)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB), trained on {len(matches)} matches")


if __name__ == "__main__":
    main()
