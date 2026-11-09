"""Mingea DFL "trădează" starea jocului? (verificare de scurgere de informație)

Pe fiecare meci DFL compară cadrele cu jocul oprit cu cele în joc:
  - cât de des lipsește mingea;
  - cât de des stă mingea complet nemișcată (deplasare 0 între cadre);
  - cât de des e mingea în afara terenului.
Dacă mingea lipsește/îngheață aproape doar când jocul e oprit, intrările de minge
sunt o scurtătură care nu există în tracking-ul nostru și trebuie scoase.

Afișează doar procente.

    E:\\mtah\\pathcrf\\.venv\\Scripts\\python.exe src\\check_dfl_ball.py --pathcrf E:\\mtah\\pathcrf
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

PITCH_X, PITCH_Y = 105.0, 68.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathcrf", required=True)
    opts = parser.parse_args()
    folder = Path(opts.pathcrf) / "data" / "sportec" / "tracking_processed"

    print(f"{'meci':<8}{'lipsă: oprit':>14}{'lipsă: joc':>12}{'nemișcată: oprit':>18}{'nemișcată: joc':>16}{'afară: oprit':>14}{'afară: joc':>12}")
    for path in sorted(folder.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["period_id", "frame_id", "ball_state", "ball_x", "ball_y"])
        df = df.sort_values(["period_id", "frame_id"]).reset_index(drop=True)
        dead = ~df["ball_state"].astype(str).str.lower().str.contains("alive").to_numpy()
        bx, by = df["ball_x"].to_numpy(float), df["ball_y"].to_numpy(float)

        missing = ~np.isfinite(bx)
        same_period = np.r_[False, df["period_id"].to_numpy()[1:] == df["period_id"].to_numpy()[:-1]]
        step = np.r_[np.nan, np.hypot(np.diff(bx), np.diff(by))]
        frozen = same_period & np.isfinite(step) & (step == 0)
        outside = np.isfinite(bx) & ((bx < 0) | (bx > PITCH_X) | (by < 0) | (by > PITCH_Y))

        def share(flag, mask):
            return f"{100 * flag[mask].mean():.1f}%"

        print(f"{path.stem:<8}{share(missing, dead):>14}{share(missing, ~dead):>12}"
              f"{share(frozen, dead):>18}{share(frozen, ~dead):>16}"
              f"{share(outside, dead):>14}{share(outside, ~dead):>12}")


if __name__ == "__main__":
    main()
