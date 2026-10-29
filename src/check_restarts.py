import pandas as pd
from pathlib import Path
import sys

# Ajustarea căii exact ca în dead_ball.py pentru a găsi modulul starter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starter.load_smart_tagging import load as load_smart_tagging

def main():
    print("=== DIAGNOSTIC RESTARTURI (V4) ===")
    
    print("Se încarcă features_ball.csv...")
    features_path = ROOT / "features_ball.csv"
    try:
        df = pd.read_csv(features_path)
    except FileNotFoundError:
        print("Eroare: features_ball.csv nu a fost găsit.")
        return

    print("Se încarcă Smart Tagging XML...")
    try:
        # Corectura principală: load() se apelează fără argumente!
        events = load_smart_tagging()
    except Exception as e:
        print(f"Eroare la încărcarea evenimentelor: {e}")
        return

    set_pieces = [e for e in events if str(e.get('code', '')).strip().upper() == 'SET PIECES']
    print(f"Total evenimente SET PIECES găsite: {len(set_pieces)}\n")

    FPS = 25
    WINDOW_FRAMES = 10 * FPS # 10 secunde înainte
    
    header = f"{'Tip SET PIECE':<22} | {'Fereastra [start, end)':<22} | {'Dur(s)':<6} | {'Minge OUT (-10s)':<17} | {'Minge LIPSĂ (-10s)':<19}"
    print(header)
    print("-" * 95)
    
    # Convertim string-urile 'True'/'False' salvate în CSV în valori logice
    df['is_valid'] = df['ball_valid'].astype(str).str.lower() == 'true'
    df['is_out'] = df['ball_outside'].astype(str).str.lower() == 'true'

    for event in set_pieces:
        start_frame = event['start_frame']
        end_frame = event['end_frame']
        duration_s = (end_frame - start_frame) / FPS
        
        # Extragem tipul exact ca în dead_ball.py
        event_type = event.get("tags", {}).get("Type", "?")
        
        # Fereastra -10 secunde
        pre_start = max(0, start_frame - WINDOW_FRAMES)
        pre_window = df[(df['frame'] >= pre_start) & (df['frame'] < start_frame)]
        
        if not pre_window.empty:
            out_count = pre_window['is_out'].sum()
            missing_count = (~pre_window['is_valid']).sum()
        else:
            out_count = 0
            missing_count = 0
            
        window_str = f"[{start_frame}, {end_frame})"
        print(f"{event_type:<22} | {window_str:<22} | {duration_s:<6.1f} | {out_count:<17} | {missing_count:<19}")

if __name__ == '__main__':
    main()