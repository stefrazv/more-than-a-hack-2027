"""Does the DFL ball give away the state of play? (information-leak check)

For each DFL match, compares dead frames with alive frames:
  - how often the ball is missing;
  - how often the ball is completely still (zero displacement between frames);
  - how often the ball is outside the pitch.
If the ball goes missing or freezes almost only when play is dead, ball inputs are a
shortcut that does not exist in the competition tracking and must be dropped.
(Result: the ball is frozen on ~100% of dead frames, so the D model uses players only.)

Prints percentages only.

    python src/check_dfl_ball.py --dfl <folder with the preprocessed DFL matches (*.parquet)>
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

PITCH_X, PITCH_Y = 105.0, 68.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl", required=True, help="folder with the preprocessed DFL matches (*.parquet)")
    opts = parser.parse_args()
    folder = Path(opts.dfl)

    print(f"{'match':<8}{'missing: dead':>14}{'missing: alive':>16}{'still: dead':>14}{'still: alive':>14}{'out: dead':>12}{'out: alive':>12}")
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

        print(f"{path.stem:<8}{share(missing, dead):>14}{share(missing, ~dead):>16}"
              f"{share(frozen, dead):>14}{share(frozen, ~dead):>14}"
              f"{share(outside, dead):>12}{share(outside, ~dead):>12}")


if __name__ == "__main__":
    main()
