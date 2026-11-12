"""
src/degrade.py
Strică tracking-ul curat DFL ca să semene cu tracking-ul automat al concursului:
jucători tăiați în bucăți puse în sloturi anonime, pauze între bucăți, dubluri,
zgomot pe poziții și goluri în minge (41%).

Rulare:  python src/degrade.py --selftest
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- Constante ---
FPS = 25
N_SLOTS = 46                # 0-22 gazde, 23-45 oaspeți
FRAG_MEAN_S = 90.0          # media lungimii unei bucăți (mediana iese ~40 s, din cauza dublurilor)
GAP_MEAN_S = 6.0            # pauza medie între două bucăți ale aceluiași jucător
DUP_PROB = 0.4              # probabilitatea ca o bucată să primească o dublură
DUP_DUR_S = 2.0
DUP_MIN_M = 0.5
DUP_MAX_M = 2.0
NOISE_PLAYER_M = 0.10
NOISE_BALL_M = 0.20
BALL_MISSING = 0.41
BALL_GAP_MEAN_S = 2.0
BALL_GAP_MAX_S = 30.0
NEAR_BALL_M = 5.0


def _player_intervals(players_xy, team, rng):
    """Toate bucățile (start, end, jucător, deplasare) -- originale și dubluri -- pe toată repriza."""
    T, num_players, _ = players_xy.shape
    frag_mean = FRAG_MEAN_S * FPS
    gap_mean = GAP_MEAN_S * FPS
    dup_mean = DUP_DUR_S * FPS
    intervals = []

    for p in range(num_players):
        present = np.isfinite(players_xy[:, p, 0])
        t = 0
        while t < T:
            while t < T and not present[t]:
                t += 1
            if t >= T:
                break
            block_end = t
            while block_end < T and present[block_end]:
                block_end += 1

            end = min(t + int(rng.exponential(frag_mean)) + 1, block_end)
            intervals.append((t, end, p, 0.0, 0.0))

            if rng.random() < DUP_PROB and end - t > 5:
                dup_start = int(rng.integers(t, end))
                dup_end = min(dup_start + int(rng.exponential(dup_mean)) + 1, end)
                dist = rng.uniform(DUP_MIN_M, DUP_MAX_M)
                angle = rng.uniform(0, 2 * np.pi)
                intervals.append((dup_start, dup_end, p, dist * np.cos(angle), dist * np.sin(angle)))

            t = end + int(rng.exponential(gap_mean))
    return intervals


def degrade(players_xy, team, ball_xy, seed=47):
    """
    players_xy: array (T, 22, 2), poziții în metri pe teren 105 x 68, NaN unde jucătorul lipsește
    team:       array (22,), 0 = gazde, 1 = oaspeți
    ball_xy:    array (T, 2), poziția mingii în metri, NaN unde lipsește
    T = cadrele UNEI reprize, 25 cadre/secundă

    returnează:
    slots_xy:   array (T, 46, 2), NaN unde slotul e gol
    slot_team:  array (46,), sloturile 0-22 = gazde, 23-45 = oaspeți
    ball_out:   array (T, 2), mingea cu goluri
    """
    rng = np.random.default_rng(seed)
    T = players_xy.shape[0]
    team = np.asarray(team)

    slot_team = np.zeros(N_SLOTS, dtype=int)
    slot_team[23:] = 1
    slots_xy = np.full((T, N_SLOTS, 2), np.nan)

    # --- Partea 1: jucătorii ---
    # Bucățile se alocă în ordinea timpului: un slot e liber dacă bucata lui anterioară s-a terminat.
    # (Alocarea jucător cu jucător fragmenta inutil sloturile și pierdea bucăți.)
    intervals = _player_intervals(players_xy, team, rng)
    order = sorted(range(len(intervals)), key=lambda i: (intervals[i][0], i))
    free_at = np.zeros(N_SLOTS, dtype=int)
    lost = 0
    for i in order:
        start, end, p, dx, dy = intervals[i]
        pool = np.flatnonzero((slot_team == team[p]) & (free_at <= start))
        if pool.size == 0:
            lost += 1
            continue
        s = rng.choice(pool)
        slots_xy[start:end, s, 0] = players_xy[start:end, p, 0] + dx
        slots_xy[start:end, s, 1] = players_xy[start:end, p, 1] + dy
        free_at[s] = end

    seen = np.isfinite(slots_xy[:, :, 0])
    slots_xy[seen] += rng.normal(0, NOISE_PLAYER_M, size=slots_xy[seen].shape)

    # --- Partea 2: mingea ---
    ball_out = ball_xy + rng.normal(0, NOISE_BALL_M, size=ball_xy.shape)

    valid = np.isfinite(players_xy[:, :, 0])
    with np.errstate(invalid="ignore"):
        dists = np.linalg.norm(players_xy - ball_xy[:, None, :], axis=-1)
    n_t = np.sum((dists <= NEAR_BALL_M) & valid, axis=1)
    prob = (1.0 + n_t) / np.sum(1.0 + n_t)

    missing = ~np.isfinite(ball_xy[:, 0])          # lipsurile deja existente se numără la țintă
    target = int(round(T * BALL_MISSING))
    gap_mean = BALL_GAP_MEAN_S * FPS
    gap_max = int(BALL_GAP_MAX_S * FPS)
    while missing.sum() < target:
        start = int(rng.choice(T, p=prob))
        if missing[start]:
            continue
        end = min(start + min(int(rng.exponential(gap_mean)) + 1, gap_max), T)
        new = np.flatnonzero(~missing[start:end]) + start
        need = target - missing.sum()
        missing[new[:need]] = True                  # ultimul gol e scurtat ca să ieșim fix la țintă
    ball_out[missing] = np.nan

    return slots_xy, slot_team, ball_out


# ---------------------------------------------------------------- test sintetic

def _synthetic_half(seed=1, T=67500):
    """22 de jucători care se mișcă lin (<= 7,5 m/s); gazdele în stânga, oaspeții în dreapta."""
    rng = np.random.default_rng(seed)
    lo = np.array([[2, 2]] * 11 + [[55, 2]] * 11, float)
    hi = np.array([[50, 66]] * 11 + [[103, 66]] * 11, float)
    pos = rng.uniform(lo, hi)
    vel = np.zeros((22, 2))
    players = np.empty((T, 22, 2))
    for t in range(T):
        vel = np.clip(0.98 * vel + rng.normal(0, 0.05, vel.shape), -0.3, 0.3)
        pos = np.clip(pos + vel, lo, hi)
        players[t] = pos
    holder = np.repeat(rng.integers(0, 22, T // (10 * FPS) + 1), 10 * FPS)[:T]
    ball = players[np.arange(T), holder]
    team = np.r_[np.zeros(11, int), np.ones(11, int)]
    return players, team, ball


def _runs(mask):
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[::2]


def selftest():
    print("Selftest degrade.py (o repriză sintetică de 45 de minute)...")
    players, team, ball = _synthetic_half()
    T = players.shape[0]
    slots, slot_team, ball_out = degrade(players, team, ball, seed=47)

    # 1. forme
    assert slots.shape == (T, N_SLOTS, 2) and slot_team.shape == (N_SLOTS,) and ball_out.shape == (T, 2)
    print("1. Forme: OK")

    # 2. niciun slot cu ambele echipe (gazdele stau în x < 52,5, oaspeții în x > 52,5)
    for s in range(N_SLOTS):
        x = slots[:, s, 0][np.isfinite(slots[:, s, 0])]
        if x.size:
            assert (x < 52.5).all() if slot_team[s] == 0 else (x > 52.5).all(), f"slotul {s} amestecă echipele"
    print("2. Fiecare slot conține o singură echipă: OK")

    # 3-4. jucători vizibili și dubluri
    for k, name in ((0, "gazde"), (1, "oaspeți")):
        vis = np.isfinite(slots[:, slot_team == k, 0]).sum(axis=1)
        over = 100 * np.mean(vis > 11)
        print(f"3-4. {name}: vizibili {vis.mean():.2f} pe cadru, peste 11 în {over:.1f}% din cadre")
        assert 10.0 <= vis.mean() <= 10.6, "media jucătorilor vizibili e în afara intervalului 10,0-10,6"
        assert 3.0 <= over <= 9.0, "procentul de cadre cu peste 11 sloturi e în afara intervalului 3-9%"

    # 5. lungimea bucăților
    lengths = np.concatenate([_runs(np.isfinite(slots[:, s, 0])) for s in range(N_SLOTS)])
    median_s = np.median(lengths) / FPS
    print(f"5. Mediana bucăților: {median_s:.1f} s")
    assert 30 <= median_s <= 55

    # 6. mingea
    missing = 100 * np.mean(~np.isfinite(ball_out[:, 0]))
    print(f"6. Mingea lipsă: {missing:.2f}%")
    assert abs(missing - 100 * BALL_MISSING) <= 0.5

    # 7. determinism
    again, _, ball_again = degrade(players, team, ball, seed=47)
    other, _, _ = degrade(players, team, ball, seed=48)
    assert np.array_equal(slots, again, equal_nan=True) and np.array_equal(ball_out, ball_again, equal_nan=True)
    assert not np.array_equal(slots, other, equal_nan=True)
    print("7. Același seed = același rezultat; alt seed = alt rezultat: OK")

    # 8. lanțul cu stitch.py
    from src.stitch import stitch_team
    for k, name in ((0, "gazde"), (1, "oaspeți")):
        tracks, _ = stitch_team(slots[:, slot_team == k])
        assert tracks.shape == (T, 11, 2) and np.isfinite(tracks).all()
        real = players[:, team == k]
        d = np.linalg.norm(tracks[:, :, None] - real[:, None], axis=-1).min(axis=2)
        print(f"8. stitch {name}: distanța la jucătorul real, mediană {np.median(d):.2f} m, p90 {np.percentile(d, 90):.2f} m")
        assert np.median(d) < 2.0

    print("Testul a trecut!")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
