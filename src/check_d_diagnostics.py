import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starter.load_smart_tagging import load as load_smart_tagging

def is_evaluated(frame):
    # Intervalele oficiale evaluate
    eval_intervals = [(11185, 90651), (110808, 190197)]
    return any(s <= frame <= e for s, e in eval_intervals)

def main():
    print("=== DIAGNOSTIC CONSERVATOR D (V4) ===")
    
    # 1. Încărcare date
    df = pd.read_csv(ROOT / "features_ball.csv")
    events = load_smart_tagging()
    set_pieces = [e for e in events if str(e.get('code', '')).strip().upper() == 'SET PIECES']
    
    # Filtrare doar pe cadrele evaluate
    df['evaluated'] = df['frame'].apply(is_evaluated)
    df_eval = df[df['evaluated']].copy()
    
    # Mască pentru ferestrele SET PIECES (inclusiv pe tot meciul)
    sp_mask = np.zeros(df['frame'].max() + 1, dtype=bool)
    for sp in set_pieces:
        sp_mask[sp['start_frame']:sp['end_frame']] = True

    # --- VERIFICAREA 1: Golurile de tracking > 2s ---
    print("\n1. GOLURI DE MINGE > 2s IN INTERVALELE EVALUATE")
    df_eval['missing'] = df_eval['ball_valid'].astype(str).str.lower() != 'true'
    m = df_eval['missing']
    
    # Grupăm secvențele consecutive lipsă
    group_id = (m != m.shift()).cumsum()
    gaps = df_eval[m].groupby(group_id).agg(
        start=('frame', 'min'), 
        end=('frame', 'max'), 
        count=('frame', 'count')
    )
    
    gaps_2s = gaps[gaps['count'] >= 50].copy() # 50 cadre = 2 secunde
    
    # Câte se intersectează cu o fereastră SET PIECE?
    gaps_2s['in_sp'] = gaps_2s.apply(lambda row: sp_mask[row['start']:row['end']+1].any(), axis=1)
    
    in_sp_count = gaps_2s['in_sp'].sum()
    out_sp_count = len(gaps_2s) - in_sp_count
    
    print(f"Total goluri > 2s identificate: {len(gaps_2s)}")
    print(f"Goluri care se suprapun cu SET PIECES: {in_sp_count}")
    print(f"Goluri în afara ferestrelor SET PIECES: {out_sp_count}")
    if out_sp_count > 0:
        print("-> CONCLUZIE: Lipsa mingii apare frecvent în joc deschis. Nu poate dicta singură starea D.")

    # --- VERIFICAREA 2: Decalajul episoadelor 'ball_outside' ---
    print("\n2. DECALAJUL IEȘIRII MINGII FAȚĂ DE ÎNCEPUTUL FERESTREI SET PIECES")
    df_eval['out'] = df_eval['ball_outside'].astype(str).str.lower() == 'true'
    o = df_eval['out']
    
    # Grupăm episoadele consecutive OUT
    group_id_out = (o != o.shift()).cumsum()
    out_episodes = df_eval[o].groupby(group_id_out).agg(
        start=('frame', 'min'), 
        end=('frame', 'max')
    )
    
    SEARCH_RADIUS = 15 * 25 # Căutăm la max 15 secunde distanță
    delays = []
    
    for sp in set_pieces:
        sp_start = sp['start_frame']
        
        # Găsim cel mai apropiat episod out de startul ferestrei
        # Distanța o calculăm față de începutul episodului OUT
        out_episodes['dist_to_sp'] = (out_episodes['start'] - sp_start).abs()
        valid_episodes = out_episodes[out_episodes['dist_to_sp'] <= SEARCH_RADIUS]
        
        if not valid_episodes.empty:
            nearest = valid_episodes.loc[valid_episodes['dist_to_sp'].idxmin()]
            # Decalaj: cu câte cadre s-a întâmplat ieșirea după (sau înainte de) startul SET PIECE
            delay = nearest['start'] - sp_start
            delays.append({
                'type': sp.get('tags', {}).get('Type', '?'),
                'delay_frames': delay,
                'delay_sec': delay / 25.0
            })

    delays_df = pd.DataFrame(delays)
    if not delays_df.empty:
        print(f"S-au putut asocia {len(delays_df)} episoade OUT la max 15s de un SET PIECE.")
        print("Statisticile decalajului (secunde = episod_out_start - fereastră_start):")
        print(delays_df.groupby('type')['delay_sec'].describe()[['count', 'mean', 'min', 'max', '50%']].round(2))
    else:
        print("Nu s-au putut asocia episoade OUT în raza de 15 secunde.")

if __name__ == '__main__':
    main()