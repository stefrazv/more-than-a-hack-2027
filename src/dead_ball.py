
# from pathlib import Path
# import sys
# import csv

# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT))

# from starter.load_smart_tagging import load

# FEATURES_PATH = ROOT / "features_ball.csv"
# FPS = 25

# # Fereastră exploratorie, nu un prag validat.
# SEARCH_RADIUS = 15 * FPS

# EVALUATED_INTERVALS = [
#     (11185, 90651),
#     (110808, 190197),
# ]


# def is_evaluated(frame):
#     return any(
#         start <= frame <= end
#         for start, end in EVALUATED_INTERVALS
#     )


# def load_outside_episodes():
#     """Grupează cadrele consecutive cu mingea în exterior."""
#     with FEATURES_PATH.open(
#         newline="", encoding="utf-8"
#     ) as file:
#         rows = list(csv.DictReader(file))

#     episodes = []
#     start = None

#     for frame, row in enumerate(rows):
#         outside = (
#             row["ball_outside"] == "True"
#             and is_evaluated(frame)
#         )

#         if outside and start is None:
#             start = frame

#         if not outside and start is not None:
#             episodes.append((start, frame - 1))
#             start = None

#     if start is not None:
#         episodes.append((start, len(rows) - 1))

#     return episodes


# def load_restarts():
#     """Selectează evenimentele SET PIECES."""
#     events = load()

#     restarts = [
#         event
#         for event in events
#         if (event["code"] or "").strip().upper()
#         == "SET PIECES"
#     ]

#     return sorted(
#         restarts,
#         key=lambda event: event["start_frame"]
#     )


# def episode_gap(event, episode):
#     """
#     Distanța dintre două intervale, în cadre.

#     0 înseamnă suprapunere.
#     """
#     event_start = event["start_frame"]
#     event_end = event["end_frame"] - 1

#     outside_start, outside_end = episode

#     if outside_end < event_start:
#         return event_start - outside_end

#     if event_end < outside_start:
#         return outside_start - event_end

#     return 0


# def main():
#     restarts = load_restarts()
#     episodes = load_outside_episodes()

#     print("=== DATE DISPONIBILE PENTRU D ===")
#     print("SET PIECES:", len(restarts))
#     print("Episoade ball_outside:", len(episodes))

#     matched = 0

#     print("\n=== PRIMELE 20 DE RELUARI ===")
#     print(
#         "Tip | Start | End | "
#         "Outside start-end | Distanta (s)"
#     )

#     for event in restarts:
#         if not episodes:
#             break

#         nearest = min(
#             episodes,
#             key=lambda ep: episode_gap(event, ep)
#         )

#         gap = episode_gap(event, nearest)

#         if gap <= SEARCH_RADIUS:
#             matched += 1

#         if restarts.index(event) >= 20:
#             continue

#         event_type = event["tags"].get("Type", "?")

#         print(
#             f"{event_type} | "
#             f"{event['start_frame']} | "
#             f"{event['end_frame']} | "
#             f"{nearest[0]}-{nearest[1]} | "
#             f"{gap / FPS:.2f}"
#         )

#     print("\nReluari cu minge in exterior")
#     print(f"la cel mult 15 secunde: {matched}/{len(restarts)}")


# def generate_dead_ball_candidates():
#     output_path = ROOT / "dead_ball_candidates.csv"

#     with FEATURES_PATH.open(
#         newline="", encoding="utf-8"
#     ) as file:
#         features = list(csv.DictReader(file))

#     restarts = load_restarts()
#     n_frames = len(features)

#     # Marcăm cadrele acoperite de cel puțin un SET PIECES.
#     set_piece_mask = [False] * n_frames

#     for event in restarts:
#         start = max(0, event["start_frame"])
#         end = min(n_frames, event["end_frame"])

#         # Intervalele Smart Tagging sunt [start, end).
#         for frame in range(start, end):
#             set_piece_mask[frame] = True

#     # Intersecția dintre cele două indicii.
#     candidates = [False] * n_frames

#     for frame, row in enumerate(features):
#         candidates[frame] = (
#             is_evaluated(frame)
#             and row["ball_outside"] == "True"
#             and set_piece_mask[frame]
#         )

#     # Numărăm episoadele consecutive candidate.
#     episodes = []
#     start = None

#     for frame, is_candidate in enumerate(candidates):
#         if is_candidate and start is None:
#             start = frame

#         if start is not None and (
#             not is_candidate or frame == n_frames - 1
#         ):
#             end = frame - 1 if not is_candidate else frame
#             episodes.append((start, end))
#             start = None

#     with output_path.open(
#         "w", newline="", encoding="utf-8"
#     ) as file:
#         writer = csv.writer(file)
#         writer.writerow(["frame", "dead_candidate"])

#         for frame, candidate in enumerate(candidates):
#             writer.writerow([frame, int(candidate)])

#     print("\n=== CANDIDATI D ===")
#     print("Cadre candidate:", sum(candidates))
#     print("Episoade candidate:", len(episodes))
#     print("Fisier generat:", output_path)

#     print("\nPrimele 10 episoade:")
#     for start, end in episodes[:10]:
#         print(
#             f"{start}-{end}: "
#             f"{(end - start + 1) / FPS:.2f} sec"
#         )

# if __name__ == "__main__":
#     main()
#     generate_dead_ball_candidates()

import pandas as pd
import numpy as np
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starter.load_smart_tagging import load as load_smart_tagging

FPS = 25

EVALUATED_INTERVALS = [
    (11185, 90651),
    (110808, 190197),
]

def is_evaluated(frame):
    return any(start <= frame <= end for start, end in EVALUATED_INTERVALS)

def main():
    print("=== GENERARE DEAD BALL V4 (CONSERVATOR) ===")
    
    # 1. Încărcare date
    features_path = ROOT / "features_ball.csv"
    df = pd.read_csv(features_path)
    events = load_smart_tagging()
    
    df['is_valid'] = df['ball_valid'].astype(str).str.lower() == 'true'
    df['is_out'] = df['ball_outside'].astype(str).str.lower() == 'true'
    df['is_eval'] = df['frame'].apply(is_evaluated)
    
    n_frames = len(df)
    is_d = np.zeros(n_frames, dtype=bool)
    
    set_pieces = [e for e in events if str(e.get('code', '')).strip().upper() == 'SET PIECES']
    
    # 2. Regula Kick-Off (Ancore perfecte de 5s)
    for sp in set_pieces:
        tag_type = sp.get('tags', {}).get('Type', '')
        if 'Kick Off' in tag_type:
            start, end = sp['start_frame'], sp['end_frame']
            # Marcăm ca D tot ce nu e minge validă în teren în această fereastră scurtă
            mask = (df['frame'] >= start) & (df['frame'] < end) & (~df['is_valid'] | df['is_out'])
            is_d[mask] = True

    # 3. Regula Ieșirilor Fizice (ball_outside + extindere conservatoare de max 5s)
    out_mask = df['is_out'] & df['is_eval']
    group_id = (out_mask != out_mask.shift()).cumsum()
    out_episodes = df[out_mask].groupby(group_id).agg(start=('frame', 'min'), end=('frame', 'max'))
    
    for _, ep in out_episodes.iterrows():
        ep_start, ep_end = int(ep['start']), int(ep['end'])
        
        # Verificăm proximitatea (± 15s) de un SET PIECE
        near_sp = any(abs(ep_start - sp['start_frame']) <= 15 * FPS for sp in set_pieces)
        
        if near_sp:
            is_d[ep_start:ep_end+1] = True # Ieșirea confirmată
            
            # Extindere plafonată spre dreapta pe lipsa de tracking (max 5s)
            frame = ep_end + 1
            added = 0
            while frame < n_frames and not df.at[frame, 'is_valid'] and added < 5 * FPS:
                is_d[frame] = True
                frame += 1
                added += 1

            # Extindere plafonată spre stânga (max 5s)
            frame = ep_start - 1
            added = 0
            while frame >= 0 and not df.at[frame, 'is_valid'] and added < 5 * FPS:
                is_d[frame] = True
                frame -= 1
                added += 1

    # 4. Regula Ancorelor Mediane (Pentru fazele fără minge ieșită, ex: Cornere invizibile)
    for sp in set_pieces:
        tag_type = sp.get('tags', {}).get('Type', '')
        anchor = None
        
        if 'Corner Kick' in tag_type:
            anchor = sp['start_frame'] + int(6.04 * FPS)
        elif 'Goal Kick' in tag_type:
            anchor = sp['start_frame'] + int(-7.36 * FPS)
            
        if anchor is not None and 0 <= anchor < n_frames:
            search_start = max(0, anchor - int(2.5 * FPS))
            search_end = min(n_frames, anchor + int(2.5 * FPS))
            
            # Dacă nu am etichetat deja zona prin ieșiri fizice...
            if not is_d[search_start:search_end].any():
                # Extindem conservator stânga-dreapta strict pe golul de minge (max 5s din ancoră)
                if not df.at[anchor, 'is_valid']:
                    frame = anchor
                    added = 0
                    while frame < n_frames and not df.at[frame, 'is_valid'] and added < 5 * FPS:
                        is_d[frame] = True
                        frame += 1
                        added += 1
                        
                    frame = anchor - 1
                    added = 0
                    while frame >= 0 and not df.at[frame, 'is_valid'] and added < 5 * FPS:
                        is_d[frame] = True
                        frame -= 1
                        added += 1

    # Pauza evaluata dinaintea kickoff-ului reprizei a doua (oficial D)
    is_d[110808:113742] = True
    
    # 5. Salvarea fișierului
    output_path = ROOT / "dead_ball_candidates.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "dead_candidate"])
        for frame in range(n_frames):
            # Salvăm D doar dacă suntem în intervalele evaluate
            final_d = is_d[frame] and df.at[frame, 'is_eval']
            writer.writerow([frame, int(final_d)])
            
    print(f"S-au etichetat {is_d.sum()} cadre ca D (candidat).")
    print(f"Fișier actualizat: {output_path}")

if __name__ == "__main__":
    main()
