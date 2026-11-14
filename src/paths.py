"""Paths used by the pipeline, in one place.

Default run (no environment variables): reads the data from ./data and writes the
intermediate files and the submission in the repository root.

For another match in the same format:
    MTAH_DATA_DIR  = folder with the data (tactical/...)
    MTAH_OUT_DIR   = folder where results are written
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import asset_paths

OUT_DIR = Path(os.environ.get("MTAH_OUT_DIR") or ROOT)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# True only for the official run on BAR-ATH, where we check the known aggregate counts.
OFFICIAL_RUN = "MTAH_OUT_DIR" not in os.environ

ATD_PATH = Path(asset_paths()["atd_raw"])
META_PATH = Path(asset_paths()["atd_meta"])

FEATURES = OUT_DIR / "features_ball.csv"
# Outputs of the rule-based version in legacy/ (not used by the submission):
DEAD = OUT_DIR / "dead_ball_candidates.csv"
POSSESSION = OUT_DIR / "possession_ha.csv"
SUBMISSION = OUT_DIR / "submission_v5.csv"
