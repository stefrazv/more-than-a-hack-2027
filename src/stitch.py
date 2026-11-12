"""Lipirea sloturilor ATD în 11 jucători continui pe echipă (pentru PathCRF).

PathCRF sare peste orice episod în care un jucător are fie și un cadru lipsă,
iar ATD-ul are 46 de sloturi care apar și dispar. Aici transformăm, pe fiecare
echipă și repriză, sloturile într-un număr fix de "jucători virtuali" fără goluri:

1. continuitate: cât timp un slot rămâne vizibil, rămâne la același jucător virtual;
2. dubluri: un slot aflat la sub DUP_DIST de un jucător deja ocupat e ignorat;
3. lipire: un slot nou primește jucătorul virtual liber cel mai apropiat de ultima
   lui poziție (cost = distanța; peste pragul 5 m + 7 m/s * timpul nevăzut,
   lipirea e doar ultima soluție și e numărată ca "forțată");
4. goluri: interpolare liniară între ultima și următoarea poziție cunoscută.

Nu folosește aleatoriu, deci rezultatul e identic la fiecare rulare.

Rulare:
    python src/stitch.py --selftest     (date sintetice, fără date din meci)
    python src/stitch.py                (pe meciul nostru; afișează doar agregate)
"""
from pathlib import Path
import argparse
import sys

import numpy as np

FPS = 25
N_TRACKS = 11
DUP_DIST = 1.0       # m: sub această distanță, două sloturi sunt același jucător
BASE_GATE = 5.0      # m: deplasare permisă imediat după dispariție
MAX_SPEED = 7.0      # m/s: viteza maximă presupusă pe durata golului
INIT_COST = 30.0     # cost pentru un jucător virtual încă nefolosit
FORCED = 1000.0      # penalizare pentru lipiturile peste prag


def _greedy_match(cost):
    """Perechi (rând, coloană) alese crescător după cost; fiecare rând/coloană o singură dată."""
    cost = cost.copy()
    pairs = []
    while np.isfinite(cost).any():
        i, k = np.unravel_index(np.argmin(cost), cost.shape)
        pairs.append((i, k, cost[i, k]))
        cost[i, :] = np.inf
        cost[:, k] = np.inf
    return pairs


def stitch_team(P, n_tracks=N_TRACKS):
    """P: (T, n_slots, 2) pozițiile sloturilor unei echipe în metri, NaN = slot gol.

    Returnează (tracks, stats): tracks are forma (T, n_tracks, 2), fără NaN.
    """
    T, n_slots, _ = P.shape
    visible = np.isfinite(P[..., 0])

    out = np.full((T, n_tracks, 2), np.nan, dtype=np.float32)
    slot_of = np.full(n_tracks, -1)          # slotul atribuit fiecărui jucător virtual
    track_of = np.full(n_slots, -1)          # jucătorul virtual atribuit fiecărui slot
    since = np.zeros(n_tracks, dtype=int)    # cadrul de la care ține slotul curent
    last_pos = np.full((n_tracks, 2), np.nan)
    last_t = np.full(n_tracks, -1)

    stats = dict(links=0, forced=0, first_use=0, duplicates=0, extras=0, frames_full=0)

    for t in range(T):
        v = visible[t]

        # 1. sloturile dispărute își eliberează jucătorul virtual
        gone = np.flatnonzero((track_of >= 0) & ~v)
        slot_of[track_of[gone]] = -1
        track_of[gone] = -1

        # 2. continuitate; dacă două sloturi continue se suprapun, îl păstrăm pe cel mai vechi
        cont = np.flatnonzero(track_of >= 0)
        if cont.size > 1:
            pos = P[t, cont]
            d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            for i, j in zip(*np.nonzero(np.triu(d < DUP_DIST, 1))):
                si, sj = cont[i], cont[j]
                if track_of[si] < 0 or track_of[sj] < 0:
                    continue
                drop = sj if since[track_of[sj]] >= since[track_of[si]] else si
                slot_of[track_of[drop]] = -1
                track_of[drop] = -1
                stats["duplicates"] += 1
            cont = np.flatnonzero(track_of >= 0)
        out[t, track_of[cont]] = P[t, cont]

        # 3. sloturi noi: dubluri, apoi lipire pe jucătorii virtuali liberi
        new = np.flatnonzero(v & (track_of < 0))
        if new.size:
            taken = out[t, slot_of >= 0]
            keep = []
            for s in new:
                ref = np.vstack([taken, P[t, keep]]) if keep else taken
                if ref.size and np.min(np.linalg.norm(ref - P[t, s], axis=1)) < DUP_DIST:
                    stats["duplicates"] += 1
                else:
                    keep.append(s)
            new = np.array(keep, dtype=int)

            free = np.flatnonzero(slot_of < 0)
            if new.size and free.size:
                cost = np.empty((new.size, free.size))
                for c, k in enumerate(free):
                    if last_t[k] < 0:
                        cost[:, c] = INIT_COST
                        continue
                    dist = np.linalg.norm(P[t, new] - last_pos[k], axis=1)
                    gate = BASE_GATE + MAX_SPEED * (t - last_t[k]) / FPS
                    cost[:, c] = np.where(dist <= gate, dist, dist + FORCED)

                for i, c, value in _greedy_match(cost):
                    s, k = new[i], free[c]
                    if last_t[k] < 0:
                        stats["first_use"] += 1
                    elif value >= FORCED:
                        stats["forced"] += 1
                    else:
                        stats["links"] += 1
                    track_of[s] = k
                    slot_of[k] = s
                    since[k] = t
                    out[t, k] = P[t, s]

            stats["extras"] += int(np.sum(track_of[new] < 0)) if new.size else 0

        active = slot_of >= 0
        last_pos[active] = out[t, active]
        last_t[active] = t
        stats["frames_full"] += int(active.all())

    # 4. golurile: interpolare liniară (la capete: ultima/prima poziție cunoscută)
    observed = np.isfinite(out[..., 0])
    gap_lengths = []
    frames = np.arange(T)
    for k in range(n_tracks):
        ok = observed[:, k]
        if not ok.any():
            out[:, k] = np.nanmean(out, axis=1)
            continue
        for axis in (0, 1):
            out[:, k, axis] = np.interp(frames, frames[ok], out[ok, k, axis])
        padded = np.concatenate([[False], ~ok, [False]])
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        gap_lengths.extend(edges[1::2] - edges[::2])

    stats["observed_share"] = float(observed.mean())
    stats["gap_median_s"] = float(np.median(gap_lengths) / FPS) if gap_lengths else 0.0
    stats["gap_max_s"] = float(np.max(gap_lengths) / FPS) if gap_lengths else 0.0
    stats["frames_full"] /= T
    return out, stats


def stitch(slots_xy, slot_team, start, end):
    """Ambele echipe, pe cadrele [start, end]. Returnează {0: (T,11,2), 1: (T,11,2)}, {echipă: stats}."""
    tracks, stats = {}, {}
    for team in (0, 1):
        cols = np.flatnonzero(np.asarray(slot_team) == team)
        tracks[team], stats[team] = stitch_team(slots_xy[start:end + 1, cols])
    return tracks, stats


# ---------------------------------------------------------------- test sintetic

def _synthetic_half(seed, T=25 * 60 * 10, n_players=11, n_slots=24):
    """11 jucători care se mișcă lin, tăiați în bucăți (medie 40 s) cu goluri (medie 8 s) și dubluri."""
    rng = np.random.default_rng(seed)
    true = np.empty((T, n_players, 2))
    pos = rng.uniform([5, 5], [100, 63], size=(n_players, 2))
    vel = np.zeros((n_players, 2))
    for t in range(T):
        vel = 0.98 * vel + rng.normal(0, 0.05, vel.shape)
        vel = np.clip(vel, -0.3, 0.3)                       # <= ~7,5 m/s pe axă
        pos = np.clip(pos + vel, [0, 0], [105, 68])
        true[t] = pos

    P = np.full((T, n_slots, 2), np.nan)
    owner = np.full((T, n_slots), -1)
    busy_until = np.zeros(n_slots, dtype=int)
    for p in range(n_players):
        t = 0
        while t < T:
            length = max(1, int(rng.exponential(40 * FPS)))
            free = np.flatnonzero(busy_until <= t)
            if free.size:
                s = free[0]
                end = min(T, t + length)
                P[t:end, s] = true[t:end, p] + rng.normal(0, 0.1, (end - t, 2))
                owner[t:end, s] = p
                busy_until[s] = end
                if rng.random() < 0.15:                      # dublură scurtă
                    free2 = np.flatnonzero(busy_until <= t)
                    if free2.size:
                        s2 = free2[0]
                        dup_end = min(end, t + 2 * FPS)
                        P[t:dup_end, s2] = true[t:dup_end, p] + 0.4
                        owner[t:dup_end, s2] = p
                        busy_until[s2] = dup_end
            t += length + max(1, int(rng.exponential(8 * FPS)))
    return true, P, owner


def selftest():
    true, P, owner = _synthetic_half(seed=47)
    tracks, stats = stitch_team(P)
    tracks2, _ = stitch_team(P)

    assert tracks.shape == (true.shape[0], N_TRACKS, 2), tracks.shape
    assert np.isfinite(tracks).all(), "au rămas NaN"
    assert np.array_equal(tracks, tracks2), "rezultatul nu e determinist"

    # cât de aproape e fiecare jucător virtual de cel mai apropiat jucător real
    d = np.linalg.norm(tracks[:, :, None] - true[:, None], axis=-1).min(axis=2)
    # identitate: pe fereastre de 10 s, ce parte din cadre urmărește același jucător real
    nearest = np.linalg.norm(tracks[:, :, None] - true[:, None], axis=-1).argmin(axis=2)
    win = 10 * FPS
    purity = []
    for a in range(0, true.shape[0] - win, win):
        block = nearest[a:a + win]
        purity.extend(np.bincount(block[:, k]).max() / win for k in range(N_TRACKS))

    print("Test sintetic: 10 minute, 11 jucători, bucăți ~40 s, goluri ~8 s, dubluri")
    print(f"  pondere poziții observate: {stats['observed_share']:.3f}")
    print(f"  lipituri: {stats['links']}, forțate: {stats['forced']}, prima folosire: {stats['first_use']}")
    print(f"  dubluri ignorate (slot-cadre): {stats['duplicates']}, sloturi în plus ignorate: {stats['extras']}")
    print(f"  distanța la cel mai apropiat jucător real: mediană {np.median(d):.2f} m, p90 {np.percentile(d, 90):.2f} m")
    print(f"  identitate stabilă pe ferestre de 10 s: {np.mean(purity):.3f}")

    assert np.median(d) < 2.0, "pozițiile lipite sunt prea departe de realitate"
    assert np.mean(purity) > 0.9, "jucătorii virtuali își schimbă identitatea prea des"
    print("Testul a trecut!")


def main():
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from src.check_slots import GRADED, load_slots
    from src.paths import ATD_PATH, META_PATH

    slots_xy, slot_team = load_slots(ATD_PATH, META_PATH)
    for half, (a, b) in enumerate(GRADED, start=1):
        _, stats = stitch(slots_xy, slot_team, a, b)
        print(f"\n=== Repriza {half} ===")
        for team, name in ((0, "gazde"), (1, "oaspeți")):
            s = stats[team]
            print(f"  {name}: observat {100 * s['observed_share']:.1f}% | toți 11 văzuți {100 * s['frames_full']:.1f}% | "
                  f"lipituri {s['links']} (forțate {s['forced']}) | dubluri (slot-cadre) {s['duplicates']} | "
                  f"în plus {s['extras']} | gol interpolat: mediană {s['gap_median_s']:.1f} s, max {s['gap_max_s']:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    if parser.parse_args().selftest:
        selftest()
    else:
        main()
