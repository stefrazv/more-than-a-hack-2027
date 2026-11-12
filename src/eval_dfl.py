"""Evaluarea PathCRF pe un meci DFL, cu metrica oficială a concursului.

Meciurile DFL au, pe fiecare cadru, starea mingii (ball_state) și echipa în posesie
(ball_owning_team_id), venite din tracking corectat -- același tip de etichete ca
ground truth-ul concursului. Din ele construim H/A/D și notăm variantele noastre cu
evaluation/possession_metrics.py, exact ca organizatorii.

Implicit: J03WR9, singurul meci DFL pe care modelul saved/140 nu l-a văzut la antrenare.

Variante:
  E1  D = adevărul DFL ("oracol"), H/A = PathCRF        -> izolează calitatea H/A
  E2  fără nicio informație de D, H/A = PathCRF peste tot
  E3  ca E2, dar cadrele în care PathCRF vede mingea ieșită devin D
Fiecare cu durata minimă a posesiei tau (o schimbare de echipă mai scurtă e ignorată).

Se rulează din repo-ul nostru, cu Python-ul mediului PathCRF:
    E:\\mtah\\pathcrf\\.venv\\Scripts\\python.exe src\\eval_dfl.py --pathcrf E:\\mtah\\pathcrf
"""
from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
TRIAL_ID = 140
TAUS = (0, 1, 2, 3, 4, 6, 8)   # secunde


# ------------------------------------------------------------------ etichete

def ground_truth(df):
    """H/A/D din coloanele DFL. Echipa gazdă = cea care deține mingea când player_id e home_*."""
    alive = df["ball_state"].astype(str).str.lower().str.contains("alive").to_numpy()
    owner = df["ball_owning_team_id"].astype("string")

    pid = df["player_id"].astype("string") if "player_id" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    home_ids = owner[pid.str.startswith("home", na=False)].mode()
    if home_ids.empty:
        raise ValueError("nu pot afla id-ul echipei gazdă")
    home_id = home_ids.iloc[0]

    is_home = owner.eq(home_id).fillna(False).to_numpy(dtype=bool)
    has_owner = owner.notna().to_numpy(dtype=bool)
    team = np.where(is_home, "H", np.where(has_owner, "A", None)).astype(object)
    team = pd.Series(team).groupby(df["period_id"].to_numpy()).transform(lambda s: s.ffill().bfill()).to_numpy()
    return np.where(alive, team, "D").astype(object), home_id


def with_half_gaps(labels, period):
    """Pune un cadru X între reprize, ca scorerul să nu lege posesii peste pauză."""
    first = np.r_[False, period[1:] != period[:-1]]
    return np.insert(labels, np.flatnonzero(first), "X")


def min_duration(labels, tau_s):
    """O posesie H/A mai scurtă de tau, lipită direct de o posesie a celeilalte echipe, revine acesteia."""
    if tau_s <= 0:
        return labels
    out = labels.copy()
    min_len = int(round(tau_s * FPS))
    i, n = 0, len(out)
    prev_team = None
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        lab = out[i]
        if lab in ("H", "A"):
            if prev_team is not None and lab != prev_team and j - i < min_len:
                out[i:j] = prev_team
                lab = prev_team
            prev_team = lab
        else:
            prev_team = None          # D sau X: după oprire posesia pornește din nou
        i = j
    return out


def score(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"], r["mean_spell_duration_s"], r["alive_dead_f1"], r["team_acc_both_alive"]


# ------------------------------------------------------------------ PathCRF

def load_model(pathcrf_dir, device):
    import torch
    from models.utils import build_model, load_trial_args, resolve_model_path
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    save_path = str(Path(pathcrf_dir) / "saved" / f"{TRIAL_ID:03d}")
    args = load_trial_args(save_path)
    model = build_model(args, device=device)
    model.load_state_dict(torch.load(resolve_model_path(save_path, "state_dict_best_acc.pt"),
                                     map_location=device, weights_only=False))
    return model, args


def pathcrf_labels(model, args, df):
    """Rulează PathCRF; întoarce (echipa H/A pe cadru, mingea ieșită pe cadru), aliniate cu df."""
    from inference import inference
    _, _, micro, _ = inference(model, df, use_crf=args.get("crf_weight", 0) > 0, decode="indep",
                               window_seconds=args.get("window_seconds", None),
                               correct_episode_lasts=False, evaluate=False)
    micro = micro.reindex(df["frame_id"].to_numpy())
    src = micro["edge_src"].astype("string")
    dst = micro["edge_dst"].astype("string")
    side = src.str.split("_").str[0]
    out = (side.eq("out") | dst.str.startswith("out", na=False)).fillna(False).to_numpy(dtype=bool)
    team = side.map({"home": "H", "away": "A"}).astype(object)
    team = pd.Series(team.to_numpy()).groupby(df["episode_id"].to_numpy()).transform(lambda s: s.ffill().bfill())
    return team.to_numpy(dtype=object), out


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathcrf", required=True)
    parser.add_argument("--match", default="J03WR9")
    parser.add_argument("--cpu", action="store_true")
    opts = parser.parse_args()

    sys.path.insert(0, str(opts.pathcrf))
    import torch
    pd.set_option("future.no_silent_downcasting", True)
    device = "cpu" if opts.cpu or not torch.cuda.is_available() else "cuda:0"

    path = Path(opts.pathcrf) / "data" / "sportec" / "tracking_processed" / f"{opts.match}.parquet"
    df = pd.read_parquet(path).sort_values(["period_id", "frame_id"]).reset_index(drop=True)
    assert df["frame_id"].is_unique, "frame_id nu e unic"
    period = df["period_id"].to_numpy()

    gt, home_id = ground_truth(df)
    gt_full = with_half_gaps(gt, period)
    print(f"Meci {opts.match}: {len(df)} cadre | D {np.mean(gt == 'D'):.1%} | gazde {np.mean(gt[gt != 'D'] == 'H'):.1%} (id {home_id})")

    model, args = load_model(opts.pathcrf, device)

    # E1: episoadele = perioadele în care mingea e în joc după DFL
    start = time.time()
    team1, _ = pathcrf_labels(model, args, df)
    print(f"E1 inferență: {time.time() - start:.0f} s")
    print(f"   acoperire pe cadrele în joc: {np.mean(pd.notna(team1[gt != 'D'])):.1%}")
    e1 = np.where(gt == "D", "D", team1)

    # E2/E3: fără informație de D -- o singură "fază în joc" pe fiecare repriză
    blind = df.copy()
    blind["ball_state"] = "alive"
    blind["episode_id"] = blind["period_id"]
    players = [c for c in blind.columns if c.startswith(("home_", "away_"))]
    blind[players] = blind.groupby("phase_id")[players].transform(lambda s: s.ffill().bfill())
    start = time.time()
    team2, out2 = pathcrf_labels(model, args, blind)
    print(f"E2 inferență: {time.time() - start:.0f} s")
    print(f"   acoperire: {np.mean(pd.notna(team2)):.1%}")
    e2 = team2.copy()
    e3 = np.where(out2, "D", team2)
    print(f"Mingea ieșită după PathCRF (E3): {out2.mean():.1%} din cadre; D real: {np.mean(gt == 'D'):.1%}")

    print(f"\n{'varianta':<34}{'spellF1':>8}{'macroF1':>9}{'posesii':>9}{'adev.':>7}{'dur(s)':>8}{'joc/oprit F1':>14}{'echipă':>8}")
    for name, pred in (("E1 D oracol + PathCRF", e1), ("E2 fără D", e2), ("E3 D din 'out' PathCRF", e3)):
        for tau in TAUS:
            p = with_half_gaps(min_duration(np.where(pd.isna(pred), "D", pred).astype(object), tau), period)
            s, m, n_pred, n_gt, dur, ad, team = score(gt_full, p)
            print(f"{name + f'  tau={tau}s':<34}{s:>8.4f}{m:>9.4f}{n_pred:>9}{n_gt:>7}{dur:>8.1f}{ad:>14.4f}{team:>8.4f}")
        print()


if __name__ == "__main__":
    main()
