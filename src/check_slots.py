"""Cât de fragmentat e tracking-ul jucătorilor? (pregătire pentru adaptorul PathCRF)

PathCRF cere, pe fiecare episod de joc, jucători prezenți pe TOATE cadrele.
ATD-ul nostru are 46 de sloturi care apar și dispar. Scriptul măsoară cât de
des și cât de lung apar sloturile, ca să știm cât de greu e să le lipim în
11 jucători continui pe echipă.

Afișează doar numere agregate (nicio dată cadru cu cadru).

Rulare:  python src/check_slots.py
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import ATD_PATH, META_PATH
from starter.baseline import team_of_each_track

FPS = 25
PITCH = (105.3, 68.0)
GRADED = [(11185, 90651), (113742, 190197)]  # reprizele, din kickoff (DATA.md / FAQ.md)


def load_slots(atd_path, meta_path):
    """ATD -> slots_xy (T, S, 2) în metri, NaN unde slotul e gol; slot_team (S,) 0=gazde, 1=oaspeți."""
    slot_team = np.asarray(team_of_each_track(meta_path), dtype=np.int8)
    n_slots = len(slot_team)

    with open(atd_path, encoding="utf-8") as f:
        n_frames = sum(1 for _ in f)

    xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    with open(atd_path, encoding="utf-8") as f:
        for t, line in enumerate(f):
            players = line.split(":")[1].split(";")
            for s, group in enumerate(players):
                x, y = group.split(",")[:2]
                xy[t, s, 0] = float(x)
                xy[t, s, 1] = float(y)

    xy[..., 0] *= PITCH[0]
    xy[..., 1] *= PITCH[1]
    return xy, slot_team


def runs(mask):
    """Lungimile secvențelor consecutive de True dintr-un vector boolean."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[::2]


def main():
    xy, slot_team = load_slots(ATD_PATH, META_PATH)
    present = np.isfinite(xy[..., 0])
    print(f"Sloturi: {len(slot_team)}  (gazde {int((slot_team == 0).sum())}, oaspeți {int((slot_team == 1).sum())})")

    for half, (a, b) in enumerate(GRADED, start=1):
        p = present[a:b + 1]
        print(f"\n=== Repriza {half}: {b - a + 1} cadre ===")

        for team, name in ((0, "gazde"), (1, "oaspeți")):
            cols = np.flatnonzero(slot_team == team)
            count = p[:, cols].sum(axis=1)
            lengths = np.concatenate([runs(p[:, c]) for c in cols])
            used = int((p[:, cols].any(axis=0)).sum())

            print(f"  {name}:")
            print(f"    sloturi folosite în repriză: {used}")
            print(f"    jucători vizibili pe cadru: medie {count.mean():.2f}, "
                  f"exact 11: {100 * (count == 11).mean():.1f}%, "
                  f"<11: {100 * (count < 11).mean():.1f}%, >11: {100 * (count > 11).mean():.1f}%")
            print(f"    bucăți de traiectorie: {len(lengths)}; lungime "
                  f"mediană {np.median(lengths) / FPS:.1f} s, "
                  f"p10 {np.percentile(lengths, 10) / FPS:.1f} s, "
                  f"p90 {np.percentile(lengths, 90) / FPS:.1f} s")

            gaps = np.concatenate([runs(~p[:, c] & (np.cumsum(p[:, c]) > 0)) for c in cols])
            if len(gaps):
                print(f"    goluri în interiorul unui slot: mediană {np.median(gaps) / FPS:.1f} s")


if __name__ == "__main__":
    main()
