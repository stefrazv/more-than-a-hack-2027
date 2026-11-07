"""Căile folosite de pipeline, într-un singur loc.

Rulare normală (fără variabile de mediu): citește datele din ./data și scrie
fișierele intermediare și submisia în rădăcina repo-ului, exact ca înainte.

Pentru alt meci în același format (ex. un meci extern convertit):
    MTAH_DATA_DIR  = folderul cu datele (tactical/...)
    MTAH_OUT_DIR   = folderul unde se scriu rezultatele
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import asset_paths

OUT_DIR = Path(os.environ.get("MTAH_OUT_DIR") or ROOT)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# True doar la rularea oficială pe BAR-ATH, unde verificăm numerele cunoscute.
OFFICIAL_RUN = "MTAH_OUT_DIR" not in os.environ

ATD_PATH = Path(asset_paths()["atd_raw"])
META_PATH = Path(asset_paths()["atd_meta"])

FEATURES = OUT_DIR / "features_ball.csv"
DEAD = OUT_DIR / "dead_ball_candidates.csv"
POSSESSION = OUT_DIR / "possession_ha.csv"
SUBMISSION = OUT_DIR / "submission_v5.csv"
