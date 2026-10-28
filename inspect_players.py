
from statistics import mean, median

frame_cautat = 20000

# Intervalele evaluate, folosind numerotarea din ATD
reprize = [
    (11186, 90652),
    (110809, 190198)
]

numar_jucatori = []
jucatori_cand_mingea_lipseste = []
jucatori_cand_mingea_exista = []

with open("data/tactical/BAR-ATH_tactical_atd.txt", "r") as f:
    for line in f:
        frame, players_text, ball_text = line.strip().split(":")
        frame = int(frame)

        # Pastram doar cadrele din reprizele evaluate
        in_meci = any(
            start <= frame <= end
            for start, end in reprize
        )

        if not in_meci and frame != frame_cautat:
            continue

        players = players_text.split(";")
        jucatori_detectati = []

        for i, player in enumerate(players, start=1):
            x, y, speed = player.split(",")

            if x != "NaN" and y != "NaN":
                jucatori_detectati.append(i)

            if frame == frame_cautat and i == 2:
                print("Slot 2 - x:", x, "y:", y, "speed:", speed)

        # Verificam cadrul 20000
        if frame == frame_cautat:
            print("Cadru analizat:", frame)
            print("Total sloturi:", len(players))
            print("Sloturi detectate:", len(jucatori_detectati))
            print("Indicii sloturilor:", jucatori_detectati)
            
            print("\n--- Echipe si minge ---")

            barcelona = [i for i in jucatori_detectati if 1 <= i <= 24]
            athletic = [i for i in jucatori_detectati if 25 <= i <= 46]

            print("Barcelona - sloturi valide:", barcelona)
            print("Athletic - sloturi valide:", athletic)

            print("Pozitia mingii:", ball_text.split(","))


        if in_meci:
            nr = len(jucatori_detectati)
            numar_jucatori.append(nr)

            ball_x, ball_y, ball_s = ball_text.split(",")

            if ball_x == "NaN" or ball_y == "NaN":
                jucatori_cand_mingea_lipseste.append(nr)
            else:
                jucatori_cand_mingea_exista.append(nr)


print("\n--- Tracking jucatori in reprizele evaluate ---")

print("Cadre analizate:", len(numar_jucatori))

print("Media sloturilor detectate:",
      round(mean(numar_jucatori), 2))

print("Mediana sloturilor detectate:",
      median(numar_jucatori))

print("Minim sloturi detectate:", min(numar_jucatori))
print("Maxim sloturi detectate:", max(numar_jucatori))

print("\n--- Comparatie cu disponibilitatea mingii ---")

print("Media jucatori cand mingea lipseste:",
      round(mean(jucatori_cand_mingea_lipseste), 2))

print("Media jucatori cand mingea exista:",
      round(mean(jucatori_cand_mingea_exista), 2))
