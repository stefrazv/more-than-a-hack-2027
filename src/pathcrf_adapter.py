"""Rulează PathCRF pe meciul nostru și combină rezultatul cu V5.

Pași:
1. ATD -> sloturi -> 11 jucători continui pe echipă (src/stitch.py);
2. tabel în formatul PathCRF (coloane home_k_x, away_k_y, ...; ball_state din V5);
3. vitezele/accelerațiile cu funcția lor (calculate_running_features);
4. inferența cu modelul pre-antrenat pe DFL (saved/140), fără evaluare;
5. H/A pe cadru = echipa jucătorului care are/pasează mingea;
6. combinare: unde V5 spune H/A, punem eticheta PathCRF; D și X rămân din V5.

Afișează doar agregate. Fișierele se scriu în pathcrf_out/ (ignorat de git).

Se rulează din repo-ul nostru, cu Python-ul mediului PathCRF:
    E:\\mtah\\pathcrf\\.venv\\Scripts\\python.exe src\\pathcrf_adapter.py --pathcrf E:\\mtah\\pathcrf
Înainte trebuie rulat o dată python src/run_all.py (are nevoie de submission_v5.csv și features_ball.csv).
"""
from pathlib import Path
import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.check_slots import FPS, GRADED, load_slots
from src.paths import ATD_PATH, FEATURES, META_PATH, OUT_DIR, SUBMISSION
from src.stitch import stitch

TRIAL_ID = 140            # Dynamic Masked CRF, cel mai bun model din lucrare
PITCH_X_OURS = 105.3      # games.py
PITCH_X_PATHCRF = 105.0   # datatools/config.py din PathCRF
OUT = OUT_DIR / "pathcrf_out"


def build_tracking(slots_xy, slot_team, v5):
    """Tabelul de intrare PathCRF, câte un rând pe cadru, doar pe reprize."""
    parts = []
    last_episode = 0
    for period, (a, b) in enumerate(GRADED, start=1):
        tracks, stats = stitch(slots_xy, slot_team, a, b)
        for team, name in ((0, "gazde"), (1, "oaspeți")):
            print(f"  R{period} {name}: observat {100 * stats[team]['observed_share']:.1f}%, "
                  f"lipituri forțate {stats[team]['forced']}")

        frames = np.arange(a, b + 1)
        alive = np.isin(v5[a:b + 1], ["H", "A"])
        starts = alive & ~np.r_[False, alive[:-1]]
        episode = np.cumsum(starts) + last_episode
        if alive.any():
            last_episode = int(episode[alive].max())

        df = pd.DataFrame({
            "frame_id": frames,
            "period_id": period,
            "timestamp": (frames - a) / FPS,
            "phase_id": period,
            "episode_id": np.where(alive, episode, 0),
            "ball_state": np.where(alive, "alive", "dead"),
            "ball_owning_team_id": np.nan,
            "ball_x": np.nan,
            "ball_y": np.nan,
        })
        cols = {}
        for team, name in ((0, "home"), (1, "away")):
            for k in range(tracks[team].shape[1]):
                cols[f"{name}_{k + 1}_x"] = tracks[team][:, k, 0] * PITCH_X_PATHCRF / PITCH_X_OURS
                cols[f"{name}_{k + 1}_y"] = tracks[team][:, k, 1]
        parts.append(pd.concat([df, pd.DataFrame(cols)], axis=1))

    return pd.concat(parts, ignore_index=True)


def run_pathcrf(tracking, pathcrf_dir, device):
    import torch
    sys.path.insert(0, str(pathcrf_dir))
    from datatools.utils import calculate_running_features
    from inference import inference
    from models.utils import build_model, load_trial_args, resolve_model_path

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tracking = calculate_running_features(tracking)

    save_path = str(Path(pathcrf_dir) / "saved" / f"{TRIAL_ID:03d}")
    args = load_trial_args(save_path)
    model = build_model(args, device=device)
    state = torch.load(resolve_model_path(save_path, "state_dict_best_acc.pt"), map_location=device, weights_only=False)
    model.load_state_dict(state)

    _, _, micro, _ = inference(
        model,
        tracking,
        use_crf=args.get("crf_weight", 0) > 0,
        decode="indep",
        window_seconds=args.get("window_seconds", None),
        correct_episode_lasts=False,
        evaluate=False,
    )
    return micro, tracking.set_index("frame_id")["episode_id"]


def spells(labels):
    """Posesii = secvențe maximale de H sau de A pe cadrele evaluate (D le întrerupe)."""
    lengths = []
    for a, b in GRADED:
        prev, run = None, 0
        for lab in labels[a:b + 1]:
            if lab in ("H", "A") and lab == prev:
                run += 1
            else:
                if prev in ("H", "A"):
                    lengths.append(run)
                prev, run = lab, 1
        if prev in ("H", "A"):
            lengths.append(run)
    return np.array(lengths)


def describe(name, labels):
    s = spells(labels)
    ha = np.isin(labels, ["H", "A"])
    home = np.mean(labels[ha] == "H")
    print(f"  {name:<10} posesii {len(s):>4} | durată medie {s.mean() / FPS:5.1f} s | "
          f"posesii <= 1 s: {100 * np.mean(s <= FPS):4.1f}% | pondere Barcelona {home:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathcrf", required=True, help="folderul repo-ului PathCRF")
    parser.add_argument("--cpu", action="store_true", help="rulează pe procesor (determinist)")
    opts = parser.parse_args()

    import torch
    device = "cpu" if opts.cpu or not torch.cuda.is_available() else "cuda:0"
    OUT.mkdir(exist_ok=True)

    v5 = pd.read_csv(SUBMISSION)["label"].to_numpy(dtype=object)

    print("1. Lipirea sloturilor...")
    slots_xy, slot_team = load_slots(ATD_PATH, META_PATH)
    tracking = build_tracking(slots_xy, slot_team, v5)

    print(f"2. PathCRF pe {device}...")
    start = time.time()
    micro, episodes = run_pathcrf(tracking, opts.pathcrf, device)
    print(f"   inferență: {time.time() - start:.0f} s")

    # 3. H/A pe cadru
    src = micro["edge_src"].astype("string")
    dst = micro["edge_dst"].astype("string")
    side = src.str.split("_").str[0]
    out_edge = side.eq("out") | dst.str.startswith("out")
    team = side.map({"home": "H", "away": "A"})
    team = team.groupby(episodes.reindex(team.index)).transform(lambda s: s.ffill().bfill())

    pathcrf = np.full(len(v5), None, dtype=object)
    pathcrf[team.index.to_numpy()] = team.to_numpy(dtype=object)
    pd.DataFrame({"frame": np.arange(len(v5)), "label": pathcrf}).to_csv(OUT / "pathcrf_ha.csv", index=False)

    alive = np.isin(v5, ["H", "A"])
    covered = alive & pd.notna(pathcrf)
    print("\n3. Agregate")
    print(f"  cadre în joc după V5: {alive.sum()} | acoperite de PathCRF: {100 * covered.sum() / alive.sum():.1f}%")
    print(f"  cadre în care PathCRF vede mingea ieșită: {100 * out_edge.reindex(np.flatnonzero(alive)).fillna(False).mean():.1f}%")
    print(f"  acord PathCRF vs V5 pe cadrele în joc: {np.mean(pathcrf[covered] == v5[covered]):.3f}")

    feat = pd.read_csv(FEATURES)
    home_d, away_d = feat["nearest_home_m"].to_numpy(), feat["nearest_away_m"].to_numpy()
    clear_h = (home_d <= 2.0) & (away_d - home_d >= 0.75)
    clear_a = (away_d <= 2.0) & (home_d - away_d >= 0.75)
    clear = covered & (clear_h | clear_a)
    evident = np.where(clear_h, "H", "A")
    print(f"  acord cu mingea vizibilă și clar la o echipă ({clear.sum()} cadre): "
          f"{np.mean(pathcrf[clear] == evident[clear]):.3f}   <- sub 0,5 înseamnă echipe inversate")

    # 4. combinare și fișierul final
    final = v5.copy()
    final[covered] = pathcrf[covered]
    out_csv = OUT / "submission_pathcrf.csv"
    pd.DataFrame({"frame": np.arange(len(final)), "label": final}).to_csv(out_csv, index=False)

    print("\n4. Structura posesiilor")
    describe("V5", v5)
    describe("PathCRF", final)

    print()
    subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(out_csv)], check=False)


if __name__ == "__main__":
    main()
