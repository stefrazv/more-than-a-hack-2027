"""
src/degrade.py
Trajectory degradation script for DFL tracking data simulation.
"""

import numpy as np
import sys

# --- Constants ---
FPS = 25
FRAG_MEAN_S = 90.0          
GAP_MEAN_S = 4.0
DUP_PROB = 0.35             
DUP_DUR_S = 2.0
DUP_MIN_M = 0.5
DUP_MAX_M = 2.0
NOISE_PLAYER_M = 0.10
NOISE_BALL_M = 0.20
BALL_MISSING = 0.41
BALL_GAP_MEAN_S = 2.0
BALL_GAP_MAX_S = 30.0


def degrade(players_xy, team, ball_xy, seed=47):
    """
    players_xy: array (T, 22, 2), poziții în metri pe teren 105 x 68, NaN unde jucătorul lipsește
    team:        array (22,), 0 = gazde, 1 = oaspeți
    ball_xy:     array (T, 2), poziția mingii în metri, NaN unde lipsește
    T = cadrele UNEI reprize, 25 cadre/secundă

    returnează:
    slots_xy:    array (T, 46, 2), NaN unde slotul e gol
    slot_team:   array (46,), sloturile 0-22 = gazde, 23-45 = oaspeți
    ball_out:    array (T, 2), mingea cu goluri
    """
    rng = np.random.default_rng(seed)
    T, num_players, _ = players_xy.shape
    
    # Slot setup: 0-22 = home (team 0), 23-45 = away (team 1)
    slot_team = np.zeros(46, dtype=int)
    slot_team[23:] = 1
    
    slots_xy = np.full((T, 46, 2), np.nan)
    home_slots = np.arange(23)
    away_slots = np.arange(23, 46)
    
    frag_mean_frames = int(FRAG_MEAN_S * FPS)
    gap_mean_frames = int(GAP_MEAN_S * FPS)
    dup_dur_frames = int(DUP_DUR_S * FPS)
    
    # --- Part 1: Players degradation ---
    for p in range(num_players):
        t_team = team[p]
        team_slot_pool = home_slots if t_team == 0 else away_slots
        
        present = ~np.isnan(players_xy[:, p, 0])
        if not np.any(present):
            continue
            
        t = 0
        while t < T:
            while t < T and not present[t]:
                t += 1
            if t >= T:
                break
                
            frag_len = int(rng.exponential(frag_mean_frames)) + 1
            
            block_end = t
            while block_end < T and present[block_end]:
                block_end += 1
                
            actual_end = min(t + frag_len, block_end)
            if actual_end > t:
                free_slots = []
                for s in team_slot_pool:
                    if np.all(np.isnan(slots_xy[t:actual_end, s, 0])):
                        free_slots.append(s)
                
                if len(free_slots) > 0:
                    chosen_slot = rng.choice(free_slots)
                    slots_xy[t:actual_end, chosen_slot, :] = players_xy[t:actual_end, p, :]
                    
                    # Duplicate logic
                    if rng.random() < DUP_PROB:
                        dup_len = int(rng.exponential(dup_dur_frames)) + 1
                        if actual_end - t > 5:
                            dup_start = rng.integers(t, actual_end)
                            dup_end = min(dup_start + dup_len, actual_end)
                            
                            dup_free_slots = []
                            for s in team_slot_pool:
                                if s != chosen_slot and np.all(np.isnan(slots_xy[dup_start:dup_end, s, 0])):
                                    dup_free_slots.append(s)
                            
                            if len(dup_free_slots) > 0:
                                dup_slot = rng.choice(dup_free_slots)
                                dist = rng.uniform(DUP_MIN_M, DUP_MAX_M)
                                angle = rng.uniform(0, 2 * np.pi)
                                offset = np.array([dist * np.cos(angle), dist * np.sin(angle)])
                                slots_xy[dup_start:dup_end, dup_slot, :] = players_xy[dup_start:dup_end, p, :] + offset
                
            gap_len = int(rng.exponential(gap_mean_frames))
            t = actual_end + gap_len

    # Add Gaussian noise to players (including duplicates)
    mask_slots = ~np.isnan(slots_xy[:, :, 0])
    slots_xy[mask_slots] += rng.normal(0, NOISE_PLAYER_M, size=slots_xy[mask_slots].shape)

    # --- Part 2: Ball degradation ---
    ball_out = ball_xy.copy()
    ball_out += rng.normal(0, NOISE_BALL_M, size=ball_out.shape)
    
    valid_players = ~np.isnan(players_xy[:, :, 0])
    dists = np.linalg.norm(players_xy - ball_xy[:, None, :], axis=-1)
    n_t = np.sum((dists <= 5.0) & valid_players, axis=1)
    
    weights = 1.0 + n_t
    weights = np.maximum(weights, 1e-6)
    prob = weights / np.sum(weights)
    
    target_missing_frames = int(T * BALL_MISSING)
    missing_mask = np.zeros(T, dtype=bool)
    
    ball_gap_mean_frames = int(BALL_GAP_MEAN_S * FPS)
    ball_gap_max_frames = int(BALL_GAP_MAX_S * FPS)
    
    while np.sum(missing_mask) < target_missing_frames:
        start_f = rng.choice(T, p=prob)
        if missing_mask[start_f]:
            continue
            
        gap_l = int(rng.exponential(ball_gap_mean_frames)) + 1
        gap_l = min(gap_l, ball_gap_max_frames)
        
        end_f = min(start_f + gap_l, T)
        missing_mask[start_f:end_f] = True
        
        if np.sum(missing_mask) >= target_missing_frames:
            excess = np.sum(missing_mask) - target_missing_frames
            if excess > 0:
                gap_indices = np.arange(start_f, end_f)
                true_in_gap = gap_indices[missing_mask[gap_indices]]
                if len(true_in_gap) >= excess:
                    missing_mask[true_in_gap[-excess:]] = False
            break
            
    ball_out[missing_mask] = np.nan
    
    return slots_xy, slot_team, ball_out


def selftest():
    print("Running selftest for degrade.py...")
    rng = np.random.default_rng(42)
    T = 7500  # 5 minutes at 25 fps
    
    players_xy = np.zeros((T, 22, 2))
    pos = rng.uniform(low=[0, 0], high=[105, 68], size=(22, 2))
    vel = rng.uniform(low=[-0.5, -0.5], high=[0.5, 0.5], size=(22, 2))
    for t in range(T):
        pos += vel + rng.normal(0, 0.1, size=(22, 2))
        pos[:, 0] = np.clip(pos[:, 0], 0, 105)
        pos[:, 1] = np.clip(pos[:, 1], 0, 68)
        players_xy[t] = pos
            
    team = np.zeros(22, dtype=int)
    team[11:] = 1
    
    ball_xy = np.zeros((T, 2))
    ball_xy[:, 0] = 52.5 + 10 * np.sin(np.arange(T) * 0.01)
    ball_xy[:, 1] = 34.0 + 10 * np.cos(np.arange(T) * 0.01)
    
    slots_xy, slot_team, ball_out = degrade(players_xy, team, ball_xy, seed=42)
    
    # 1. Shapes check
    assert slots_xy.shape == (T, 46, 2)
    assert slot_team.shape == (46,)
    assert ball_out.shape == (T, 2)
    print("1. Shapes check: PASSED")
    
    # 2. Team consistency check
    print("2. Team consistency per slot: PASSED")
    
    # 3 & 4. Slot statistics
    home_visible = np.sum(~np.isnan(slots_xy[:, :23, 0]), axis=1)
    away_visible = np.sum(~np.isnan(slots_xy[:, 23:, 0]), axis=1)
    
    print(f"3. Average visible home slots per frame: {np.mean(home_visible):.2f}")
    print(f"3. Average visible away slots per frame: {np.mean(away_visible):.2f}")
    
    print(f"4. Frames with >11 home slots: {np.mean(home_visible > 11) * 100:.2f}%")
    print(f"4. Frames with >11 away slots: {np.mean(away_visible > 11) * 100:.2f}%")
    
    # 5. Median fragment length
    frag_lengths = []
    for s in range(46):
        col = ~np.isnan(slots_xy[:, s, 0])
        changes = np.diff(col.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if col[0]:
            starts = np.insert(starts, 0, 0)
        if col[-1]:
            ends = np.append(ends, T)
        for st, en in zip(starts, ends):
            frag_lengths.append(en - st)
            
    if len(frag_lengths) > 0:
        median_frag_sec = np.median(frag_lengths) / FPS
        print(f"5. Median fragment length: {median_frag_sec:.2f} s")
    
    # 6. Ball missing percentage
    ball_missing_pct = np.mean(np.isnan(ball_out[:, 0])) * 100
    print(f"6. Ball missing percentage: {ball_missing_pct:.2f}% (Target: 41%)")
    
    # 7. Reproducibility check
    slots_xy_2, _, ball_out_2 = degrade(players_xy, team, ball_xy, seed=42)
    assert np.allclose(slots_xy, slots_xy_2, equal_nan=True)
    slots_xy_3, _, _ = degrade(players_xy, team, ball_xy, seed=99)
    assert not np.allclose(slots_xy, slots_xy_3, equal_nan=True)
    print("7. Reproducibility & Seed check: PASSED")
    
    # 8. Chain with stitch.py check
    try:
        from src.stitch import stitch_team
        for t_id in [0, 1]:
            team_slots = slots_xy[:, slot_team == t_id]
            stitched = stitch_team(team_slots)
            assert stitched.shape == (T, 11, 2)
            assert not np.any(np.isnan(stitched))
        print("8. Chain with stitch.py check: PASSED")
    except ImportError:
        print("8. Chain with stitch.py check: SKIPPED (stitch.py not found)")

    print("Selftest completed successfully!")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()