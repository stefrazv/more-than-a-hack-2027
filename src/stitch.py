"""Stitch the ATD slots into 11 continuous players per team (input for PathCRF).

PathCRF skips any episode in which a player misses even one frame, while the ATD has
46 slots that appear and disappear. For each team and half, we turn the slots into a
fixed number of gap-free "virtual players":

1. continuity: while a slot stays visible, it stays with the same virtual player;
2. duplicates: a slot closer than DUP_DIST to an already assigned player is ignored;
3. joining: a new slot takes the free virtual player nearest to its last position
   (cost = distance; beyond 5 m + 7 m/s * time unseen, the join is a last resort
   and is counted as "forced");
4. gaps: linear interpolation between the last and the next known position.

No randomness, so the result is identical on every run.

    python src/stitch.py --selftest     (synthetic data, no match data)
    python src/stitch.py                (on the competition match; prints aggregates only)
"""
from pathlib import Path
import argparse
import sys

import numpy as np

FPS = 25
N_TRACKS = 11
DUP_DIST = 1.0       # m: below this distance, two slots are the same player
BASE_GATE = 5.0      # m: displacement allowed right after disappearing
MAX_SPEED = 7.0      # m/s: maximum speed assumed during the gap
INIT_COST = 30.0     # cost of a virtual player not used yet
FORCED = 1000.0      # penalty for joins beyond the gate


def _greedy_match(cost):
    """(row, column) pairs picked by increasing cost; each row and column used once."""
    cost = cost.copy()
    pairs = []
    while np.isfinite(cost).any():
        i, k = np.unravel_index(np.argmin(cost), cost.shape)
        pairs.append((i, k, cost[i, k]))
        cost[i, :] = np.inf
        cost[:, k] = np.inf
    return pairs


def stitch_team(P, n_tracks=N_TRACKS):
    """P: (T, n_slots, 2) slot positions of one team in metres, NaN = empty slot.

    Returns (tracks, stats): tracks has shape (T, n_tracks, 2), with no NaN.
    """
    T, n_slots, _ = P.shape
    visible = np.isfinite(P[..., 0])

    out = np.full((T, n_tracks, 2), np.nan, dtype=np.float32)
    slot_of = np.full(n_tracks, -1)          # slot assigned to each virtual player
    track_of = np.full(n_slots, -1)          # virtual player assigned to each slot
    since = np.zeros(n_tracks, dtype=int)    # frame since which the current slot is held
    last_pos = np.full((n_tracks, 2), np.nan)
    last_t = np.full(n_tracks, -1)

    stats = dict(links=0, forced=0, first_use=0, duplicates=0, extras=0, frames_full=0)

    for t in range(T):
        v = visible[t]

        # 1. slots that disappeared release their virtual player
        gone = np.flatnonzero((track_of >= 0) & ~v)
        slot_of[track_of[gone]] = -1
        track_of[gone] = -1

        # 2. continuity; if two continuing slots overlap, keep the older one
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

        # 3. new slots: drop duplicates, then join them to the free virtual players
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

    # 4. gaps: linear interpolation (at the ends: last/first known position)
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
    """Both teams, on frames [start, end]. Returns {0: (T,11,2), 1: (T,11,2)}, {team: stats}."""
    tracks, stats = {}, {}
    for team in (0, 1):
        cols = np.flatnonzero(np.asarray(slot_team) == team)
        tracks[team], stats[team] = stitch_team(slots_xy[start:end + 1, cols])
    return tracks, stats


# ---------------------------------------------------------------- synthetic test

def _synthetic_half(seed, T=25 * 60 * 10, n_players=11, n_slots=24):
    """11 smoothly moving players, cut into fragments (mean 40 s) with gaps (mean 8 s) and duplicates."""
    rng = np.random.default_rng(seed)
    true = np.empty((T, n_players, 2))
    pos = rng.uniform([5, 5], [100, 63], size=(n_players, 2))
    vel = np.zeros((n_players, 2))
    for t in range(T):
        vel = 0.98 * vel + rng.normal(0, 0.05, vel.shape)
        vel = np.clip(vel, -0.3, 0.3)                       # <= ~7.5 m/s per axis
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
                if rng.random() < 0.15:                      # short duplicate
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
    assert np.isfinite(tracks).all(), "NaN left"
    assert np.array_equal(tracks, tracks2), "result is not deterministic"

    # how close each virtual player is to the nearest real player
    d = np.linalg.norm(tracks[:, :, None] - true[:, None], axis=-1).min(axis=2)
    # identity: in 10 s windows, share of frames that follow the same real player
    nearest = np.linalg.norm(tracks[:, :, None] - true[:, None], axis=-1).argmin(axis=2)
    win = 10 * FPS
    purity = []
    for a in range(0, true.shape[0] - win, win):
        block = nearest[a:a + win]
        purity.extend(np.bincount(block[:, k]).max() / win for k in range(N_TRACKS))

    print("Synthetic test: 10 minutes, 11 players, fragments ~40 s, gaps ~8 s, duplicates")
    print(f"  share of observed positions: {stats['observed_share']:.3f}")
    print(f"  joins: {stats['links']}, forced: {stats['forced']}, first use: {stats['first_use']}")
    print(f"  duplicates ignored (slot-frames): {stats['duplicates']}, extra slots ignored: {stats['extras']}")
    print(f"  distance to the nearest real player: median {np.median(d):.2f} m, p90 {np.percentile(d, 90):.2f} m")
    print(f"  identity stable over 10 s windows: {np.mean(purity):.3f}")

    assert np.median(d) < 2.0, "stitched positions are too far from the real ones"
    assert np.mean(purity) > 0.9, "virtual players change identity too often"
    print("Test passed.")


def main():
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from src.check_slots import GRADED, load_slots
    from src.paths import ATD_PATH, META_PATH

    slots_xy, slot_team = load_slots(ATD_PATH, META_PATH)
    for half, (a, b) in enumerate(GRADED, start=1):
        _, stats = stitch(slots_xy, slot_team, a, b)
        print(f"\n=== Half {half} ===")
        for team, name in ((0, "home"), (1, "away")):
            s = stats[team]
            print(f"  {name}: observed {100 * s['observed_share']:.1f}% | all 11 seen {100 * s['frames_full']:.1f}% | "
                  f"joins {s['links']} (forced {s['forced']}) | duplicates (slot-frames) {s['duplicates']} | "
                  f"extra {s['extras']} | interpolated gap: median {s['gap_median_s']:.1f} s, max {s['gap_max_s']:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    if parser.parse_args().selftest:
        selftest()
    else:
        main()
